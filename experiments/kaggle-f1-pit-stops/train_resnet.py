"""Train a residual MLP on the engineered features and cache OOF + test predictions."""
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from features import build_features

warnings.filterwarnings('ignore')

SEED = 42
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
OOF_DIR = Path('oof'); OOF_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        h = self.dropout(torch.relu(self.fc1(x)))
        h = self.fc2(h)
        return torch.relu(self.bn(h + x))


class ResNetMLP(nn.Module):
    def __init__(self, in_dim, hidden=192, dropout=0.25, n_blocks=4):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.dropout(torch.relu(self.input(x)))
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


def train_fold(X_tr, y_tr, X_va, y_va, X_te, epochs=40, patience=6):
    model = ResNetMLP(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=4096, shuffle=True, drop_last=True,
    )
    X_va_t = torch.tensor(X_va).to(DEVICE)
    X_te_t = torch.tensor(X_te).to(DEVICE)

    best_auc = 0.0
    best_oof = best_pred = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()

        model.eval()
        with torch.no_grad():
            oof_p = torch.sigmoid(model(X_va_t)).cpu().numpy()
            te_p  = torch.sigmoid(model(X_te_t)).cpu().numpy()
        auc = roc_auc_score(y_va, oof_p)

        improved = auc > best_auc
        if improved:
            best_auc, best_oof, best_pred = auc, oof_p.copy(), te_p.copy()
            no_improve = 0
        else:
            no_improve += 1
        print(f'  ep {epoch+1:02d}  loss={running/len(loader):.4f}  val_auc={auc:.5f}' + ('  *' if improved else ''))
        if no_improve >= patience:
            print(f'  early stop at epoch {epoch+1}')
            break

    return best_oof, best_pred, best_auc


def main():
    print(f'device: {DEVICE}')
    X, X_te, y, feature_cols, folds, _ = build_features(for_nn=True)
    print(f'X={X.shape}  features={len(feature_cols)}')

    oof = np.zeros(len(X), dtype=np.float32)
    pred = np.zeros(len(X_te), dtype=np.float32)
    fold_aucs = []

    for f, (tr_idx, va_idx) in enumerate(folds):
        t0 = time.time()
        print(f'\nfold {f+1}/{len(folds)}')
        oof_f, pred_f, auc_f = train_fold(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], X_te)
        oof[va_idx] = oof_f
        pred += pred_f / len(folds)
        fold_aucs.append(auc_f)
        print(f'  fold {f+1}: AUC {auc_f:.5f}  ({time.time()-t0:.0f}s)')

    overall = roc_auc_score(y, oof)
    print(f'\nOOF AUC: {overall:.5f}')
    print('fold AUCs:', [round(a, 5) for a in fold_aucs])
    np.savez(OOF_DIR / 'resnet.npz', oof=oof, pred=pred)


if __name__ == '__main__':
    main()
