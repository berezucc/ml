"""Optuna search over CatBoost hyperparameters on fold 0.

Each trial trains one CatBoost on the fold-0 split. ~5 min per trial. The
best params are saved to `oof/cb_best_params.json` and can be plugged into
the bagged CatBoost section of the notebook.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from features import CAT_COLS, build_features

warnings.filterwarnings('ignore')
SEED = 42
OOF_DIR = Path('oof'); OOF_DIR.mkdir(exist_ok=True)


def objective(trial, X_tr, y_tr, X_va, y_va):
    params = dict(
        iterations=trial.suggest_int('iterations', 1500, 5000, step=500),
        depth=trial.suggest_int('depth', 5, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.02, 0.10, log=True),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0),
        random_strength=trial.suggest_float('random_strength', 0.5, 3.0),
        eval_metric='AUC',
        cat_features=CAT_COLS,
        random_seed=SEED,
        early_stopping_rounds=100,
        verbose=0,
        task_type='CPU',
    )
    m = CatBoostClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    return roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=10800,
                        help='wall-clock budget in seconds (default 3 hours)')
    args = parser.parse_args()

    X, _, y, _, folds, _ = build_features()
    X_cb = X.copy()
    for c in CAT_COLS:
        X_cb[c] = X_cb[c].astype(str).fillna('NA')
    tr, va = folds[0]
    X_tr, X_va, y_tr, y_va = X_cb.iloc[tr], X_cb.iloc[va], y[tr], y[va]
    print(f'tuning on fold 0: {len(tr)} train / {len(va)} val')

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    t0 = time.time()
    study.optimize(
        lambda t: objective(t, X_tr, y_tr, X_va, y_va),
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=False,
    )
    print(f'\nbest AUC: {study.best_value:.5f}  ({time.time()-t0:.0f}s)')
    print('best params:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    with (OOF_DIR / 'cb_best_params.json').open('w') as f:
        json.dump({'auc': study.best_value, 'params': study.best_params}, f, indent=2)
    print(f'\nsaved {OOF_DIR}/cb_best_params.json')


if __name__ == '__main__':
    main()
