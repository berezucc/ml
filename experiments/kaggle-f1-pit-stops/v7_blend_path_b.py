"""
v7 — Path B blend: 4-model stacker (LGB + bagged CB + XGB + ResNet) blended
with Anthony Therrien's public NN-Residual submission via logit-rank.

Why both paths: Path A alone uses Anthony's output (LB 0.95453); adding our
4-model stacker at a small weight (0.05) ties at 0.95452 — confirming the
ResNet adds diversity but the LB ceiling without insider "vault" submissions
sits at ~0.954.

Requires:
  - oof/lgb.npz, oof/cb_bag.npz, oof/xgb.npz   (from notebook)
  - oof/resnet.npz                              (from train_resnet.py)
  - external/anthony/submission.csv             (from `kaggle kernels output`)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


DATA    = Path('data')
OOF     = Path('oof')
ANTHONY = Path('external/anthony/submission.csv')
WEIGHT  = 0.05  # final blend weight on our stacker; Anthony anchors


def to_rank(x, eps=1e-6):
    r = rankdata(x) / len(x)
    return np.clip(r, eps, 1 - eps)


def logit_rank_blend(anchor: np.ndarray, support: np.ndarray, w: float) -> np.ndarray:
    z = (1 - w) * logit(to_rank(anchor)) + w * logit(to_rank(support))
    blended_rank = expit(z)
    order = np.argsort(blended_rank, kind='mergesort')
    out = np.empty_like(anchor, dtype=float)
    out[order] = np.sort(anchor)
    return np.clip(out, 1e-7, 1 - 1e-7)


def main():
    # 4-model stacker on our OOFs
    names = ['lgb', 'cb_bag', 'xgb', 'resnet']
    oofs  = {n: np.load(OOF / f'{n}.npz')['oof']  for n in names}
    preds = {n: np.load(OOF / f'{n}.npz')['pred'] for n in names}

    tr = pd.read_csv(DATA / 'train.csv')
    tr_sorted = tr.sort_values(['Race', 'Year', 'Driver', 'LapNumber']).reset_index(drop=True)
    y = tr_sorted['PitNextLap'].astype(int).values

    aucs = {n: roc_auc_score(y, oofs[n]) for n in names}
    print('single OOF AUCs:', {n: round(a, 5) for n, a in aucs.items()})

    stack_X    = np.column_stack([logit(to_rank(oofs[n]))  for n in names])
    stack_X_te = np.column_stack([logit(to_rank(preds[n])) for n in names])
    lr = LogisticRegression(C=10, max_iter=500).fit(stack_X, y)
    print('LR coefs:', dict(zip(names, lr.coef_[0].round(4))))

    stack_oof  = lr.predict_proba(stack_X)[:, 1]
    stack_pred = lr.predict_proba(stack_X_te)[:, 1]
    print(f'stacker OOF AUC: {roc_auc_score(y, stack_oof):.5f}')

    # Align test predictions to original id order
    te = pd.read_csv(DATA / 'test.csv')
    te_sorted = te.sort_values(['Race', 'Year', 'Driver', 'LapNumber']).reset_index(drop=True)
    test_ids_sorted = te_sorted['id'].values
    ours = pd.DataFrame({'id': test_ids_sorted, 'PitNextLap': stack_pred}).set_index('id').sort_index()

    # Anthony anchor
    if not ANTHONY.exists():
        raise SystemExit(
            f'Missing {ANTHONY}. Run:\n'
            '  kaggle kernels output anthonytherrien/predicting-f1-pit-stops-nn-residual-network -p ./external/anthony'
        )
    ant = pd.read_csv(ANTHONY).set_index('id').sort_index()
    assert (ant.index == ours.index).all()

    final = logit_rank_blend(ant['PitNextLap'].values, ours['PitNextLap'].values, WEIGHT)

    # Restore original test id order
    out = pd.DataFrame({'id': ant.index.values, 'PitNextLap': final})
    test_order = te[['id']]
    out = test_order.merge(out, on='id', how='left')
    out.to_csv('submission.csv', index=False)
    print(f'wrote submission.csv  weight={WEIGHT}  rows={len(out)}  mean={out.PitNextLap.mean():.5f}')


if __name__ == '__main__':
    main()
