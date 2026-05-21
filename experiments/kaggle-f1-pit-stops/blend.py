"""Blend cached OOFs with an optional external anchor submission and write submission.csv.

Usage:
    python blend.py                  # 4-model logit-rank blend only
    python blend.py --anchor PATH    # blend our LR stacker with the anchor
    python blend.py --weight 0.05    # control anchor blend weight
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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


def stacker(oofs, preds, y):
    names = list(oofs)
    X_oof = np.column_stack([logit(to_rank(oofs[n])) for n in names])
    X_te  = np.column_stack([logit(to_rank(preds[n])) for n in names])
    lr = LogisticRegression(C=10, max_iter=500).fit(X_oof, y)
    coefs = dict(zip(names, lr.coef_[0].round(4)))
    return lr.predict_proba(X_oof)[:, 1], lr.predict_proba(X_te)[:, 1], coefs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anchor', type=Path, default=None,
                        help='external submission.csv to use as logit-rank anchor')
    parser.add_argument('--weight', type=float, default=0.05,
                        help='weight assigned to our stacker when anchored')
    parser.add_argument('--out', type=Path, default=Path('submission.csv'))
    args = parser.parse_args()

    names = ['lgb', 'cb_bag', 'xgb', 'resnet']
    oofs, preds = load_cached_oofs(names)

    train = pd.read_csv(DATA_DIR / 'train.csv')
    y = (train.sort_values(['Race', 'Year', 'Driver', 'LapNumber'])[TARGET]
              .reset_index(drop=True).astype(int).values)

    for n in names:
        print(f'  {n:<8} OOF AUC {roc_auc_score(y, oofs[n]):.5f}')

    stack_oof, stack_pred, coefs = stacker(oofs, preds, y)
    print('LR coefs:', coefs)
    print(f'stacker OOF AUC: {roc_auc_score(y, stack_oof):.5f}')

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
