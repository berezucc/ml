"""Shared experiment code for TabFM vs CatBoost on the S6E5 pit-stop data.

Every model run is cached as artifacts/runs/<key>.json, so the driver
(run_all.py) and the notebook can both call these functions and only the
first caller pays the compute cost.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE.parent / 'kaggle-f1-pit-stops'
DATA_DIR = BASELINE_DIR / 'data'
ARTIFACTS = HERE / 'artifacts'
RUNS = ARTIFACTS / 'runs'
RUNS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASELINE_DIR))

SEED = 42
FOLD = 0
N_EVAL = 5_000        # fixed eval subsample of the fold-0 validation split
N_CTX = 5_000         # TabFM context rows, raw features
N_CTX_FE = 2_000      # engineered features: TabFM cost scales ~linearly with
                      # column count (104 vs 14 cols), so the budget is smaller
SWEEP_CTX = [500, 1_000, 2_000, 5_000, 10_000]
N_ESTIMATORS = 8      # TabFM ensemble members (default 32; 8 is ~4x faster,
                      # near-identical AUC on this data)
RERUN = False

_state = {}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def data():
    """Load raw + engineered features in the exact row order of features.py."""
    if _state:
        return _state
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    for df in (train, test):
        df.rename(columns={'LapTime (s)': 'LapTime'}, inplace=True)

    # reproduce features.py row order exactly (concat with test, sort, keep train)
    train['is_test'] = 0
    test['is_test'] = 1
    test['PitNextLap'] = np.nan
    full = (pd.concat([train, test], ignore_index=True)
              .sort_values(['Race', 'Year', 'Driver', 'LapNumber'])
              .reset_index(drop=True))
    train_sorted = full[full['is_test'] == 0].reset_index(drop=True)

    from features import CAT_COLS as fe_cat
    raw_cols = [c for c in train_sorted.columns if c not in ('id', 'PitNextLap', 'is_test')]
    X_raw = train_sorted[raw_cols].copy()
    y = train_sorted['PitNextLap'].astype(int).values

    fe_cache = ARTIFACTS / 'train_fe.parquet'
    lf_cache = ARTIFACTS / 'labels_folds.npz'
    if fe_cache.exists() and lf_cache.exists():
        X_fe = pd.read_parquet(fe_cache)
        cached_lf = np.load(lf_cache)
        y_fe, fold_id = cached_lf['y'], cached_lf['fold_id']
        folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    else:
        from features import build_features
        X_fe, _, y_fe, _, folds, _ = build_features(data_dir=str(DATA_DIR))
        for c in fe_cat:
            X_fe[c] = X_fe[c].astype(str)
        X_fe.to_parquet(fe_cache)
        fid = np.full(len(y_fe), -1)
        for f, (_, va) in enumerate(folds):
            fid[va] = f
        np.savez(lf_cache, y=y_fe, fold_id=fid)
    assert np.array_equal(y, y_fe), 'raw / engineered row order mismatch'

    tr_idx, va_idx = folds[FOLD]
    eval_idx = np.random.default_rng(SEED).choice(va_idx, size=N_EVAL, replace=False)

    with open(BASELINE_DIR / 'oof' / 'cb_best_params.json') as f:
        cb_tuned = json.load(f)['params']

    _state.update(
        X_raw=X_raw, X_fe=X_fe, y=y, tr_idx=tr_idx, va_idx=va_idx,
        eval_idx=eval_idx, raw_cat=['Driver', 'Compound', 'Race'],
        fe_cat=list(fe_cat), cb_tuned=cb_tuned,
    )
    log(f'data ready: raw {X_raw.shape}, engineered {X_fe.shape} ({time.time()-t0:.0f}s)')
    return _state


def sample_ctx(n):
    d = data()
    return np.random.default_rng(SEED).choice(d['tr_idx'], size=n, replace=False)


_tabfm = {}


def tabfm_model():
    if not _tabfm:
        import torch
        from tabfm import tabfm_v1_0_0_pytorch as v1
        device = 'mps' if torch.backends.mps.is_available() else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        t0 = time.time()
        _tabfm['model'] = v1.load().to(device)
        log(f'TabFM loaded on {device} ({time.time()-t0:.0f}s)')
    return _tabfm['model']


def cached(key, fn):
    path = RUNS / f'{key}.json'
    if path.exists() and not RERUN:
        return json.loads(path.read_text())
    log(f'run {key} ...')
    result = fn()
    path.write_text(json.dumps(result))
    log(f'run {key}: AUC {result["auc"]:.4f}  {result["seconds"]:,.0f}s')
    return result


def run_tabfm(feats, n_ctx, chunk=2_000):
    def fn():
        from sklearn.metrics import roc_auc_score
        from tabfm import TabFMClassifier
        d = data()
        X = d['X_raw'] if feats == 'raw' else d['X_fe']
        ctx, ev, y = sample_ctx(n_ctx), d['eval_idx'], d['y']
        clf = TabFMClassifier(model=tabfm_model(), n_estimators=N_ESTIMATORS,
                              random_state=SEED)
        t0 = time.time()
        clf.fit(X.iloc[ctx], y[ctx])
        proba = np.concatenate([
            clf.predict_proba(X.iloc[ev[i:i + chunk]])[:, 1]
            for i in range(0, len(ev), chunk)
        ])
        return dict(model='TabFM', features=feats, train_rows=n_ctx,
                    auc=roc_auc_score(y[ev], proba), seconds=time.time() - t0)
    return cached(f'tabfm_{feats}_{n_ctx}', fn)


def run_catboost(feats, n_rows=None, tuned=False):
    """n_rows=None -> full fold-0 training split."""
    def fn():
        from catboost import CatBoostClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        d = data()
        X = d['X_raw'] if feats == 'raw' else d['X_fe']
        cat_cols = d['raw_cat'] if feats == 'raw' else d['fe_cat']
        fit_idx = d['tr_idx'] if n_rows is None else sample_ctx(n_rows)
        ev, y = d['eval_idx'], d['y']

        Xc = X.copy()
        for c in cat_cols:
            Xc[c] = Xc[c].astype(str).fillna('NA')
        f_tr, f_va = train_test_split(fit_idx, test_size=0.1, random_state=SEED,
                                      stratify=y[fit_idx])
        params = d['cb_tuned'] if tuned else dict(iterations=1000, learning_rate=0.08, depth=8)
        m = CatBoostClassifier(**params, eval_metric='AUC', cat_features=cat_cols,
                               random_seed=SEED, early_stopping_rounds=100,
                               verbose=0, task_type='CPU')
        t0 = time.time()
        m.fit(Xc.iloc[f_tr], y[f_tr], eval_set=(Xc.iloc[f_va], y[f_va]),
              use_best_model=True)
        proba = m.predict_proba(Xc.iloc[ev])[:, 1]
        return dict(model='CatBoost', features=feats,
                    train_rows=len(fit_idx), auc=roc_auc_score(y[ev], proba),
                    seconds=time.time() - t0)
    n = n_rows if n_rows is not None else 'full'
    suffix = '_tuned' if tuned else ''
    return cached(f'cb_{feats}_{n}{suffix}', fn)


def grid_df():
    """Experiment A: model x features grid on the shared eval sample."""
    rows = [
        run_tabfm('raw', N_CTX),
        run_tabfm('engineered', N_CTX_FE),
        run_catboost('raw', N_CTX),
        run_catboost('engineered', N_CTX_FE),
        run_catboost('raw'),
        run_catboost('engineered', tuned=True),
    ]
    return pd.DataFrame(rows)


def sweep_df():
    """Experiment B: same-n-rows sweep, raw features."""
    rows = []
    for n in SWEEP_CTX:
        rows.append({**run_tabfm('raw', n), 'n_rows': n})
        rows.append({**run_catboost('raw', n), 'n_rows': n})
    return pd.DataFrame(rows)
