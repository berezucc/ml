"""Blend cached OOFs with an optional external anchor submission and write submission.csv.

Usage:
    python blend.py                          # LR stacker, no anchor
    python blend.py --stacker gbdt           # LightGBM stacker instead of LR
    python blend.py --anchor PATH            # blend stacker with the anchor
    python blend.py --anchor PATH --weight 0.05
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import TARGET, build_features

OOF_DIR = Path('oof')
DATA_DIR = Path('data')


def to_rank(x, eps=1e-6):
    return np.clip(rankdata(x) / len(x), eps, 1 - eps)


def logit_rank(anchor, support, weight):
    z = (1 - weight) * logit(to_rank(anchor)) + weight * logit(to_rank(support))
    blended = expit(z)
    order = np.argsort(blended, kind='mergesort')
    out = np.empty_like(anchor, dtype=float)
    out[order] = np.sort(anchor)
    return np.clip(out, 1e-7, 1 - 1e-7)


def load_cached_oofs(names):
    oofs = {n: np.load(OOF_DIR / f'{n}.npz')['oof'] for n in names}
    preds = {n: np.load(OOF_DIR / f'{n}.npz')['pred'] for n in names}
    return oofs, preds


def lr_stacker(oofs, preds, y):
    names = list(oofs)
    X_oof = np.column_stack([logit(to_rank(oofs[n])) for n in names])
    X_te  = np.column_stack([logit(to_rank(preds[n])) for n in names])
    lr = LogisticRegression(C=10, max_iter=500).fit(X_oof, y)
    return lr.predict_proba(X_oof)[:, 1], lr.predict_proba(X_te)[:, 1], dict(zip(names, lr.coef_[0].round(4)))


def gbdt_stacker(oofs, preds, y, n_folds=5, seed=42):
    """Stack with a small LightGBM. OOF predictions over 5 internal folds."""
    names = list(oofs)
    X_oof = np.column_stack([logit(to_rank(oofs[n])) for n in names])
    X_te  = np.column_stack([logit(to_rank(preds[n])) for n in names])
    oof_pred = np.zeros(len(y))
    te_pred  = np.zeros(len(X_te))
    params = dict(
        objective='binary', metric='auc',
        learning_rate=0.02, num_leaves=15, min_child_samples=200,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
        lambda_l2=1.0, verbose=-1, seed=seed, n_jobs=-1,
    )
    for tr, va in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X_oof, y):
        m = lgb.train(
            params,
            lgb.Dataset(X_oof[tr], y[tr]),
            500,
            valid_sets=[lgb.Dataset(X_oof[va], y[va])],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        oof_pred[va] = m.predict(X_oof[va], num_iteration=m.best_iteration)
        te_pred += m.predict(X_te, num_iteration=m.best_iteration) / n_folds
    importance = dict(zip(names, m.feature_importance(importance_type='gain').round(1)))
    return oof_pred, te_pred, importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anchor', type=Path, default=None,
                        help='external submission.csv to use as logit-rank anchor')
    parser.add_argument('--weight', type=float, default=0.05,
                        help='weight assigned to our stacker when anchored')
    parser.add_argument('--stacker', choices=['lr', 'gbdt'], default='lr')
    parser.add_argument('--out', type=Path, default=Path('submission.csv'))
    args = parser.parse_args()

    names = ['lgb', 'cb_bag', 'xgb', 'resnet']
    oofs, preds = load_cached_oofs(names)

    train = pd.read_csv(DATA_DIR / 'train.csv')
    y = (train.sort_values(['Race', 'Year', 'Driver', 'LapNumber'])[TARGET]
              .reset_index(drop=True).astype(int).values)

    for n in names:
        print(f'  {n:<8} OOF AUC {roc_auc_score(y, oofs[n]):.5f}')

    if args.stacker == 'lr':
        stack_oof, stack_pred, meta = lr_stacker(oofs, preds, y)
        print(f'LR coefs: {meta}')
    else:
        stack_oof, stack_pred, meta = gbdt_stacker(oofs, preds, y)
        print(f'GBDT gain importance: {meta}')
    print(f'{args.stacker} stacker OOF AUC: {roc_auc_score(y, stack_oof):.5f}')

    test = pd.read_csv(DATA_DIR / 'test.csv')
    sorted_ids = (test.sort_values(['Race', 'Year', 'Driver', 'LapNumber'])
                      .reset_index(drop=True)['id'].values)
    ours = (pd.DataFrame({'id': sorted_ids, TARGET: stack_pred})
              .set_index('id').sort_index())

    if args.anchor is None:
        final = stack_pred
        id_order = sorted_ids
    else:
        if not args.anchor.exists():
            raise SystemExit(f'anchor file not found: {args.anchor}')
        anchor = pd.read_csv(args.anchor).set_index('id').sort_index()
        assert anchor.index.equals(ours.index), 'anchor and test ids do not match'
        final = logit_rank(anchor[TARGET].values, ours[TARGET].values, args.weight)
        id_order = anchor.index.values

    out = (pd.DataFrame({'id': id_order, TARGET: final})
             .merge(test[['id']], on='id', how='right'))
    out.to_csv(args.out, index=False)
    print(f'wrote {args.out}  rows={len(out)}  mean={out[TARGET].mean():.5f}')


if __name__ == '__main__':
    main()
