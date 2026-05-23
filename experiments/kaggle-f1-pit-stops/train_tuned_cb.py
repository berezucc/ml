"""Train CatBoost on all folds using params from `oof/cb_best_params.json`
(produced by `tune_catboost.py`). Saves to `oof/cb_tuned.npz`.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from features import CAT_COLS, build_features

warnings.filterwarnings('ignore')
SEED = 42
OOF = Path('oof'); OOF.mkdir(exist_ok=True)


def main():
    with (OOF / 'cb_best_params.json').open() as f:
        meta = json.load(f)
    params = meta['params']
    print('using params:', params)

    X, X_te, y, _, folds, _ = build_features()
    X_cb = X.copy()
    X_te_cb = X_te.copy()
    for c in CAT_COLS:
        X_cb[c]    = X_cb[c].astype(str).fillna('NA')
        X_te_cb[c] = X_te_cb[c].astype(str).fillna('NA')

    oof = np.zeros(len(X))
    pred = np.zeros(len(X_te))
    fold_aucs = []
    for f, (tr, va) in enumerate(folds):
        t0 = time.time()
        m = CatBoostClassifier(
            **params,
            eval_metric='AUC', cat_features=CAT_COLS, random_seed=SEED + f,
            early_stopping_rounds=120, verbose=0, task_type='CPU',
        )
        m.fit(X_cb.iloc[tr], y[tr], eval_set=(X_cb.iloc[va], y[va]), use_best_model=True)
        oof[va] = m.predict_proba(X_cb.iloc[va])[:, 1]
        pred += m.predict_proba(X_te_cb)[:, 1] / len(folds)
        fold_aucs.append(roc_auc_score(y[va], oof[va]))
        print(f'  fold {f+1}: AUC {fold_aucs[-1]:.5f}  iter {m.tree_count_}  {time.time()-t0:.0f}s')

    print(f'\ntuned CB OOF AUC: {roc_auc_score(y, oof):.5f}')
    print('fold AUCs:', [round(a, 5) for a in fold_aucs])
    np.savez(OOF / 'cb_tuned.npz', oof=oof, pred=pred)


if __name__ == '__main__':
    main()
