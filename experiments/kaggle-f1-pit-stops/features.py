"""Feature engineering and OOF target encoding for the F1 pit-stop dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


TARGET = 'PitNextLap'
ID = 'id'
CAT_COLS = [
    'Driver', 'Compound', 'Race',
    'Compound_prev', 'Compound_next',
    'Compound_prev2', 'Compound_next2',
]
TE_GROUPS = [
    ('Driver',),
    ('Race',),
    ('Compound', 'Stint'),
    ('Driver', 'Compound'),
    ('Race', 'Compound'),
    ('Driver', 'Race'),
]
TE_ALPHA = 20.0


def _add_lag_lead(full, group):
    g = full.groupby(group, sort=False)
    one_step_cols = [
        'TyreLife', 'LapTime', 'Position', 'Stint', 'Compound',
        'PitStop', 'Cumulative_Degradation', 'LapTime_Delta',
        'Position_Change', 'LapNumber',
    ]
    two_step_cols = ['TyreLife', 'LapTime', 'Stint', 'Compound', 'PitStop', 'LapNumber']
    for col in one_step_cols:
        full[f'{col}_prev'] = g[col].shift(1)
        full[f'{col}_next'] = g[col].shift(-1)
    for col in two_step_cols:
        full[f'{col}_prev2'] = g[col].shift(2)
        full[f'{col}_next2'] = g[col].shift(-2)

    full['lap_gap_prev']  = full['LapNumber'] - full['LapNumber_prev']
    full['lap_gap_next']  = full['LapNumber_next']  - full['LapNumber']
    full['lap_gap_next2'] = full['LapNumber_next2'] - full['LapNumber']
    full['lap_gap_prev2'] = full['LapNumber'] - full['LapNumber_prev2']

    full['tyre_life_delta_next']  = full['TyreLife_next']  - full['TyreLife']
    full['tyre_life_delta_next2'] = full['TyreLife_next2'] - full['TyreLife']
    full['tyre_life_delta_prev']  = full['TyreLife'] - full['TyreLife_prev']

    full['stint_change_next']     = (full['Stint_next']  > full['Stint']).astype(float)
    full['stint_change_next2']    = (full['Stint_next2'] > full['Stint']).astype(float)
    full['stint_change_prev']     = (full['Stint'] > full['Stint_prev']).astype(float)
    full['compound_change_next']  = (full['Compound_next']  != full['Compound']).astype(float)
    full['compound_change_next2'] = (full['Compound_next2'] != full['Compound']).astype(float)
    full['just_pitted']           = (full['stint_change_prev'] == 1).astype(float)

    # When the next observed lap is exactly N+1, PitStop_next is the target.
    full['oracle_pit_next'] = np.where(full['lap_gap_next'] == 1, full['PitStop_next'], np.nan)
    full['oracle_gap2_pit'] = np.where(
        (full['lap_gap_next'] == 2) & (full['Stint_next'] > full['Stint']),
        1.0, np.nan,
    )

    full['laptime_diff_prev']  = full['LapTime'] - full['LapTime_prev']
    full['laptime_diff_next']  = full['LapTime_next'] - full['LapTime']
    full['laptime_diff_prev2'] = full['LapTime'] - full['LapTime_prev2']
    full['pos_diff_prev']      = full['Position'] - full['Position_prev']
    full['pos_diff_next']      = full['Position_next'] - full['Position']
    full['laptime_roll3_mean'] = g['LapTime'].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    full['laptime_vs_recent'] = full['LapTime'] - full['laptime_roll3_mean']


def _add_stint(full, stint_group):
    gs = full.groupby(stint_group, sort=False)
    full['stint_lap_min']    = gs['LapNumber'].transform('min')
    full['stint_lap_max']    = gs['LapNumber'].transform('max')
    full['stint_lap_count']  = gs['LapNumber'].transform('count')
    full['stint_tyre_max']   = gs['TyreLife'].transform('max')
    full['stint_progress']   = (full['LapNumber'] - full['stint_lap_min']) / (full['stint_lap_max'] - full['stint_lap_min'] + 1)
    full['laps_left_in_stint'] = full['stint_lap_max'] - full['LapNumber']
    full['tyre_left_in_stint'] = full['stint_tyre_max'] - full['TyreLife']
    full['stint_laptime_mean'] = gs['LapTime'].transform('mean')
    full['laptime_vs_stint']   = full['LapTime'] - full['stint_laptime_mean']
    full['stint_deg_max']      = gs['Cumulative_Degradation'].transform('max')
    full['deg_pct_of_max']     = full['Cumulative_Degradation'] / (full['stint_deg_max'] + 1e-6)
    full['is_last_obs_of_stint'] = (full['LapNumber'] == full['stint_lap_max']).astype(float)


def _add_compound_norm(full):
    # Rebuild a Normalized_TyreLife-like signal from per-compound quantiles.
    tr = full['is_test'] == 0
    cm = full[tr].groupby('Compound')['TyreLife'].quantile(0.95).rename('compound_typical_max')
    full = full.merge(cm, on='Compound', how='left')
    full['tyre_life_normalised'] = full['TyreLife'] / (full['compound_typical_max'] + 1e-6)
    full['tyre_life_over_max']   = (full['TyreLife'] > full['compound_typical_max']).astype(float)
    csl = full[tr].groupby('Compound')['stint_lap_count'].mean().rename('compound_avg_stint_len')
    full = full.merge(csl, on='Compound', how='left')
    full['stint_len_vs_compound_avg'] = full['stint_lap_count'] - full['compound_avg_stint_len']
    return full


def _add_race_context(full, race_group):
    rl = full.groupby(['Race', 'Year', 'LapNumber'])
    full['n_drivers_this_lap']     = rl['Driver'].transform('count')
    full['mean_position_this_lap'] = rl['Position'].transform('mean')
    full['mean_laptime_this_lap']  = rl['LapTime'].transform('mean')
    full['laptime_vs_field']       = full['LapTime'] - full['mean_laptime_this_lap']
    full['field_pitrate_this_lap'] = rl['PitStop'].transform('mean')

    rlc = full.groupby(['Race', 'Year', 'LapNumber', 'Compound'])
    full['n_same_compound_this_lap'] = rlc['Driver'].transform('count')
    full['frac_same_compound']       = full['n_same_compound_this_lap'] / full['n_drivers_this_lap']

    ru = (full.groupby(['Race', 'Year', 'LapNumber'])['PitStop']
          .sum()
          .reset_index()
          .rename(columns={'PitStop': 'race_pits_this_lap'})
          .sort_values(['Race', 'Year', 'LapNumber']))
    ru['race_pits_next_lap'] = ru.groupby(['Race', 'Year'])['race_pits_this_lap'].shift(-1)
    full = full.merge(
        ru[['Race', 'Year', 'LapNumber', 'race_pits_next_lap']],
        on=['Race', 'Year', 'LapNumber'], how='left',
    )

    full['stints_done_so_far'] = full['Stint'] - 1
    full['race_total_laps']    = full.groupby(race_group)['LapNumber'].transform('max')
    full['laps_left_in_race']  = full['race_total_laps'] - full['LapNumber']
    full['frac_laps_left']     = full['laps_left_in_race'] / (full['race_total_laps'] + 1e-6)
    return full


def _add_driver_pits(full, group):
    g = full.groupby(group, sort=False)
    full['driver_pits_so_far'] = g['PitStop'].cumsum() - full['PitStop']


def _target_encode(train_fe, test_fe, y, folds, global_mean):
    for cols in TE_GROUPS:
        cols = list(cols)
        key = 'te_' + '_x_'.join(cols)
        oof = np.full(len(train_fe), global_mean, dtype=np.float32)
        for tr_idx, va_idx in folds:
            sub = train_fe.iloc[tr_idx][cols].copy()
            sub['_y'] = y[tr_idx]
            agg = sub.groupby(cols, observed=True)['_y'].agg(['sum', 'count'])
            agg['te'] = (agg['sum'] + TE_ALPHA * global_mean) / (agg['count'] + TE_ALPHA)
            merged = train_fe.iloc[va_idx][cols].merge(
                agg.reset_index()[cols + ['te']], on=cols, how='left',
            )
            oof[va_idx] = merged['te'].fillna(global_mean).astype(np.float32).values

        sub = train_fe[cols].copy(); sub['_y'] = y
        agg = sub.groupby(cols, observed=True)['_y'].agg(['sum', 'count'])
        agg['te'] = (agg['sum'] + TE_ALPHA * global_mean) / (agg['count'] + TE_ALPHA)
        te_pred = (test_fe[cols]
                   .merge(agg.reset_index()[cols + ['te']], on=cols, how='left')['te']
                   .fillna(global_mean).astype(np.float32).values)

        train_fe[key] = oof
        test_fe[key] = te_pred


def build_features(data_dir='data', n_folds=5, seed=42, for_nn=False):
    """Run the full FE + OOF target-encoding pipeline.

    If ``for_nn`` is True, returns numpy arrays with label-encoded categoricals
    and standardised + clipped numerics, ready for a neural net. Otherwise
    returns pandas DataFrames with category dtypes preserved for GBDTs.
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / 'train.csv')
    test  = pd.read_csv(data_dir / 'test.csv')
    for df in (train, test):
        df.rename(columns={'LapTime (s)': 'LapTime'}, inplace=True)

    train['is_test'] = 0
    test['is_test'] = 1
    test[TARGET] = np.nan
    full = (pd.concat([train, test], ignore_index=True)
              .sort_values(['Race', 'Year', 'Driver', 'LapNumber'])
              .reset_index(drop=True))

    group = ['Race', 'Year', 'Driver']
    stint_group = group + ['Stint']
    race_group = ['Race', 'Year']

    _add_lag_lead(full, group)
    _add_stint(full, stint_group)
    full = _add_compound_norm(full)
    full = _add_race_context(full, race_group)
    _add_driver_pits(full, group)

    for col in CAT_COLS:
        full[col] = full[col].astype('category')

    train_fe = full[full['is_test'] == 0].copy()
    test_fe  = full[full['is_test'] == 1].copy()

    y = train_fe[TARGET].astype(int).values
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(train_fe, y))
    _target_encode(train_fe, test_fe, y, folds, global_mean=y.mean())

    feature_cols = [c for c in train_fe.columns if c not in (ID, TARGET, 'is_test')]
    test_ids_sorted = test_fe[ID].values

    if not for_nn:
        return train_fe[feature_cols], test_fe[feature_cols], y, feature_cols, folds, test_ids_sorted

    # NN prep: label-encode cats, fill na, scale, clip
    X    = train_fe[feature_cols].copy()
    X_te = test_fe[feature_cols].copy()
    for col in CAT_COLS:
        le = LabelEncoder().fit(pd.concat([X[col].astype(str), X_te[col].astype(str)]))
        X[col]    = le.transform(X[col].astype(str)).astype(np.float32)
        X_te[col] = le.transform(X_te[col].astype(str)).astype(np.float32)
    for col in feature_cols:
        if X[col].isnull().any() or X_te[col].isnull().any():
            med = X[col].median()
            X[col] = X[col].fillna(med)
            X_te[col] = X_te[col].fillna(med)
    scaler = StandardScaler()
    X_arr    = np.clip(scaler.fit_transform(X.astype(np.float64)), -10, 10).astype(np.float32)
    X_te_arr = np.clip(scaler.transform(X_te.astype(np.float64)), -10, 10).astype(np.float32)
    return X_arr, X_te_arr, y, feature_cols, folds, test_ids_sorted
