"""
B2 — Residual MLP on our 104 engineered features.

Architecture inspired by Anthony Therrien's notebook (3 residual blocks,
hidden=96, dropout=0.2) but adapted for binary classification:
  - BCEWithLogitsLoss instead of L1 regression
  - AUC-based early stopping
  - Trained per-fold on our engineered features (incl. TE + oracle)
  - Saves OOF + test predictions to oof/resnet.npz for blending

Run after the main notebook has produced oof/{lgb,cb_bag,xgb}.npz
(no dependency on those; just shares folds + features).
"""
from pathlib import Path
import time, warnings, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')
SEED, N_FOLDS = 42, 5
DATA  = Path('data')
OOF   = Path('oof'); OOF.mkdir(exist_ok=True)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'device: {DEVICE}')
torch.manual_seed(SEED)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        r = x
        x = self.drop(self.act(self.l1(x)))
        x = self.l2(x)
        x = self.bn(x + r)
        return self.act(x)


class ResNetMLP(nn.Module):
    def __init__(self, in_dim, hidden=96, dropout=0.2, n_blocks=3):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.drop(self.act(self.input(x)))
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


def build_features():
    """Reproduce the v5 FE + TE pipeline. Returns (X, X_te, y, feature_cols, CAT_COLS, folds)."""
    train = pd.read_csv(DATA / 'train.csv')
    test  = pd.read_csv(DATA / 'test.csv')
    train.rename(columns={'LapTime (s)': 'LapTime'}, inplace=True)
    test.rename(columns={'LapTime (s)': 'LapTime'}, inplace=True)
    TARGET, ID = 'PitNextLap', 'id'
    train['is_test'] = 0; test['is_test'] = 1; test[TARGET] = np.nan
    full = pd.concat([train, test], ignore_index=True)
    full = full.sort_values(['Race','Year','Driver','LapNumber']).reset_index(drop=True)
    G  = ['Race','Year','Driver']
    GS = ['Race','Year','Driver','Stint']
    RACE = ['Race','Year']

    g = full.groupby(G, sort=False)
    for col in ['TyreLife','LapTime','Position','Stint','Compound','PitStop',
                'Cumulative_Degradation','LapTime_Delta','Position_Change','LapNumber']:
        full[f'{col}_prev'] = g[col].shift(1)
        full[f'{col}_next'] = g[col].shift(-1)
    for col in ['TyreLife','LapTime','Stint','Compound','PitStop','LapNumber']:
        full[f'{col}_prev2'] = g[col].shift(2)
        full[f'{col}_next2'] = g[col].shift(-2)
    full['lap_gap_prev']  = full['LapNumber'] - full['LapNumber_prev']
    full['lap_gap_next']  = full['LapNumber_next']  - full['LapNumber']
    full['lap_gap_next2'] = full['LapNumber_next2'] - full['LapNumber']
    full['lap_gap_prev2'] = full['LapNumber'] - full['LapNumber_prev2']
    full['tyre_life_delta_next']  = full['TyreLife_next']  - full['TyreLife']
    full['tyre_life_delta_next2'] = full['TyreLife_next2'] - full['TyreLife']
    full['tyre_life_delta_prev']  = full['TyreLife']       - full['TyreLife_prev']
    full['stint_change_next']     = (full['Stint_next']  > full['Stint']).astype('float')
    full['stint_change_next2']    = (full['Stint_next2'] > full['Stint']).astype('float')
    full['stint_change_prev']     = (full['Stint'] > full['Stint_prev']).astype('float')
    full['compound_change_next']  = (full['Compound_next']  != full['Compound']).astype('float')
    full['compound_change_next2'] = (full['Compound_next2'] != full['Compound']).astype('float')
    full['just_pitted']           = (full['stint_change_prev'] == 1).astype('float')
    full['oracle_pit_next'] = np.where(full['lap_gap_next'] == 1, full['PitStop_next'], np.nan)
    full['oracle_gap2_pit'] = np.where(
        (full['lap_gap_next'] == 2) & (full['Stint_next'] > full['Stint']), 1.0, np.nan)
    full['laptime_diff_prev']  = full['LapTime'] - full['LapTime_prev']
    full['laptime_diff_next']  = full['LapTime_next'] - full['LapTime']
    full['laptime_diff_prev2'] = full['LapTime'] - full['LapTime_prev2']
    full['pos_diff_prev']      = full['Position'] - full['Position_prev']
    full['pos_diff_next']      = full['Position_next'] - full['Position']
    full['laptime_roll3_mean'] = g['LapTime'].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    full['laptime_vs_recent']  = full['LapTime'] - full['laptime_roll3_mean']

    gs = full.groupby(GS, sort=False)
    full['stint_lap_min']      = gs['LapNumber'].transform('min')
    full['stint_lap_max']      = gs['LapNumber'].transform('max')
    full['stint_lap_count']    = gs['LapNumber'].transform('count')
    full['stint_tyre_max']     = gs['TyreLife'].transform('max')
    full['stint_progress']     = (full['LapNumber'] - full['stint_lap_min']) / (full['stint_lap_max'] - full['stint_lap_min'] + 1)
    full['laps_left_in_stint'] = full['stint_lap_max'] - full['LapNumber']
    full['tyre_left_in_stint'] = full['stint_tyre_max'] - full['TyreLife']
    full['stint_laptime_mean'] = gs['LapTime'].transform('mean')
    full['laptime_vs_stint']   = full['LapTime'] - full['stint_laptime_mean']
    full['stint_deg_max']      = gs['Cumulative_Degradation'].transform('max')
    full['deg_pct_of_max']     = full['Cumulative_Degradation'] / (full['stint_deg_max'] + 1e-6)
    full['is_last_obs_of_stint'] = (full['LapNumber'] == full['stint_lap_max']).astype('float')

    tr_mask = full['is_test'] == 0
    cm = full[tr_mask].groupby('Compound')['TyreLife'].quantile(0.95).rename('compound_typical_max')
    full = full.merge(cm, on='Compound', how='left')
    full['tyre_life_normalised'] = full['TyreLife'] / (full['compound_typical_max'] + 1e-6)
    full['tyre_life_over_max']   = (full['TyreLife'] > full['compound_typical_max']).astype('float')
    csl = full[tr_mask].groupby('Compound')['stint_lap_count'].mean().rename('compound_avg_stint_len')
    full = full.merge(csl, on='Compound', how='left')
    full['stint_len_vs_compound_avg'] = full['stint_lap_count'] - full['compound_avg_stint_len']

    rl = full.groupby(['Race','Year','LapNumber'])
    full['n_drivers_this_lap']     = rl['Driver'].transform('count')
    full['mean_position_this_lap'] = rl['Position'].transform('mean')
    full['mean_laptime_this_lap']  = rl['LapTime'].transform('mean')
    full['laptime_vs_field']       = full['LapTime'] - full['mean_laptime_this_lap']
    full['field_pitrate_this_lap'] = rl['PitStop'].transform('mean')
    rlc = full.groupby(['Race','Year','LapNumber','Compound'])
    full['n_same_compound_this_lap'] = rlc['Driver'].transform('count')
    full['frac_same_compound']       = full['n_same_compound_this_lap'] / full['n_drivers_this_lap']
    ru = full.groupby(['Race','Year','LapNumber'])['PitStop'].sum().reset_index().rename(columns={'PitStop':'race_pits_this_lap'})
    ru = ru.sort_values(['Race','Year','LapNumber'])
    ru['race_pits_next_lap'] = ru.groupby(['Race','Year'])['race_pits_this_lap'].shift(-1)
    full = full.merge(ru[['Race','Year','LapNumber','race_pits_next_lap']], on=['Race','Year','LapNumber'], how='left')
    full['stints_done_so_far'] = full['Stint'] - 1
    full['race_total_laps']    = full.groupby(RACE)['LapNumber'].transform('max')
    full['laps_left_in_race']  = full['race_total_laps'] - full['LapNumber']
    full['frac_laps_left']     = full['laps_left_in_race'] / (full['race_total_laps'] + 1e-6)
    full['driver_pits_so_far'] = g['PitStop'].cumsum() - full['PitStop']

    CAT_COLS = ['Driver','Compound','Race','Compound_prev','Compound_next','Compound_prev2','Compound_next2']
    for c in CAT_COLS:
        full[c] = full[c].astype('category')

    train_fe = full[full['is_test']==0].copy()
    test_fe  = full[full['is_test']==1].copy()

    # TE features
    TE_GROUPS = [('Driver',), ('Race',), ('Compound','Stint'),
                 ('Driver','Compound'), ('Race','Compound'), ('Driver','Race')]
    ALPHA = 20.0
    y = train_fe[TARGET].astype(int).values
    gmean = y.mean()
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train_fe, y))

    for grp in TE_GROUPS:
        cols = list(grp); key = 'te_' + '_x_'.join(cols)
        oof = np.full(len(train_fe), gmean, dtype=np.float32)
        for tr, va in folds:
            sub = train_fe.iloc[tr][cols].copy(); sub['_y'] = y[tr]
            agg = sub.groupby(cols, observed=True)['_y'].agg(['sum','count'])
            agg['te'] = (agg['sum'] + ALPHA * gmean) / (agg['count'] + ALPHA)
            m = train_fe.iloc[va][cols].merge(agg.reset_index()[cols+['te']], on=cols, how='left')
            oof[va] = m['te'].fillna(gmean).astype(np.float32).values
        sub = train_fe[cols].copy(); sub['_y'] = y
        agg = sub.groupby(cols, observed=True)['_y'].agg(['sum','count'])
        agg['te'] = (agg['sum'] + ALPHA * gmean) / (agg['count'] + ALPHA)
        te_pred = test_fe[cols].merge(agg.reset_index()[cols+['te']], on=cols, how='left')['te'].fillna(gmean).astype(np.float32).values
        train_fe[key] = oof; test_fe[key] = te_pred

    drop_cols = [ID, TARGET, 'is_test']
    feature_cols = [c for c in train_fe.columns if c not in drop_cols]

    # Prep for NN: label-encode cats, fill na, scale, clip
    X    = train_fe[feature_cols].copy()
    X_te = test_fe[feature_cols].copy()
    for c in CAT_COLS:
        le = LabelEncoder()
        le.fit(pd.concat([X[c].astype(str), X_te[c].astype(str)]))
        X[c]    = le.transform(X[c].astype(str)).astype(np.float32)
        X_te[c] = le.transform(X_te[c].astype(str)).astype(np.float32)
    for c in feature_cols:
        if X[c].isnull().any() or X_te[c].isnull().any():
            med = X[c].median()
            X[c] = X[c].fillna(med); X_te[c] = X_te[c].fillna(med)
    scaler = StandardScaler()
    X    = np.clip(scaler.fit_transform(X.astype(np.float64)), -10, 10).astype(np.float32)
    X_te = np.clip(scaler.transform(X_te.astype(np.float64)), -10, 10).astype(np.float32)
    return X, X_te, y, feature_cols, CAT_COLS, folds, test_fe[ID].values


def train_one_fold(X_tr, y_tr, X_va, y_va, X_te, in_dim, epochs=40):
    model = ResNetMLP(in_dim, hidden=192, dropout=0.25, n_blocks=4).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32)
    Xva_t = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)
    yva   = y_va
    Xte_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=4096, shuffle=True, drop_last=True)

    best_auc, best_oof, best_pred, patience = 0.0, None, None, 6
    no_improve = 0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        model.eval()
        with torch.no_grad():
            oof_pred = torch.sigmoid(model(Xva_t)).cpu().numpy()
            te_pred  = torch.sigmoid(model(Xte_t)).cpu().numpy()
        auc = roc_auc_score(yva, oof_pred)
        marker = ''
        if auc > best_auc:
            best_auc, best_oof, best_pred = auc, oof_pred.copy(), te_pred.copy()
            no_improve = 0; marker = '  *'
        else:
            no_improve += 1
        print(f'    ep {ep+1:02d}  train_loss={tot/len(loader):.4f}  val_auc={auc:.5f}{marker}')
        if no_improve >= patience:
            print(f'    early-stop @ ep {ep+1}')
            break
    return best_oof, best_pred, best_auc


def main():
    print('building features...')
    X, X_te, y, feature_cols, _, folds, _ = build_features()
    print(f'X={X.shape}  X_te={X_te.shape}  features={len(feature_cols)}')

    oof  = np.zeros(len(X), dtype=np.float32)
    pred = np.zeros(len(X_te), dtype=np.float32)
    fold_aucs = []
    for f, (tr, va) in enumerate(folds):
        t0 = time.time()
        print(f'\nfold {f+1}/{N_FOLDS} ({len(tr)} train / {len(va)} val):')
        oof_f, pred_f, auc_f = train_one_fold(X[tr], y[tr], X[va], y[va], X_te, X.shape[1])
        oof[va] = oof_f
        pred   += pred_f / N_FOLDS
        fold_aucs.append(auc_f)
        print(f'  fold {f+1} best AUC: {auc_f:.5f}  ({time.time()-t0:.1f}s)')

    full_auc = roc_auc_score(y, oof)
    print(f'\nResNet OOF AUC: {full_auc:.5f}')
    print('fold AUCs:', [round(a, 5) for a in fold_aucs])
    np.savez(OOF / 'resnet.npz', oof=oof, pred=pred)
    print(f'saved oof/resnet.npz')


if __name__ == '__main__':
    main()
