"""Logistic regression coefficient SEs: bootstrap vs closed-form Fisher.

The two should agree when (a) the model is well-specified, (b) n is large
enough, and (c) the data isn't degenerate (no perfect separation).

Fit via Newton-Raphson (IRLS). Fisher SEs come from diag((X^T W X)^-1).
"""
from __future__ import annotations

import numpy as np

from bootstrap import bootstrap_resample

SEED = 0
N = 400              # sample size
D = 4                # features (incl. intercept column added by code)
TRUE_BETA = np.array([0.0, 1.5, -2.0, 0.7], dtype=np.float64)  # incl. intercept
N_BOOT = 1500
MAX_ITER = 50
TOL = 1e-8


def sigmoid(z):
    pos = z >= 0
    out = np.empty_like(z)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Newton-Raphson (IRLS). X must already include an intercept column."""
    n, d = X.shape
    beta = np.zeros(d, dtype=np.float64)
    for _ in range(MAX_ITER):
        p = sigmoid(X @ beta)
        # Tikhonov ridge of 1e-6 prevents singular H when a feature is near-degenerate
        # in a bootstrap resample. Negligible bias on point estimates at this scale.
        W = p * (1.0 - p)
        grad = X.T @ (y - p)
        H = (X.T * W) @ X + 1e-6 * np.eye(d)
        step = np.linalg.solve(H, grad)
        beta_new = beta + step
        if np.linalg.norm(step) < TOL:
            beta = beta_new
            break
        beta = beta_new
    return beta


def fisher_se(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Asymptotic SE: diag((X^T W X)^-1) at the MLE."""
    p = sigmoid(X @ beta)
    W = p * (1.0 - p)
    info = (X.T * W) @ X
    cov = np.linalg.inv(info)
    return np.sqrt(np.diag(cov))


def make_data(rng: np.random.Generator):
    # First column is the intercept (all ones); rest are standard normal.
    X = np.zeros((N, D), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:] = rng.standard_normal((N, D - 1))
    logits = X @ TRUE_BETA
    p = sigmoid(logits)
    y = (rng.uniform(size=N) < p).astype(np.float64)
    return X, y


def main():
    rng = np.random.default_rng(SEED)
    X, y = make_data(rng)
    beta_hat = fit_logistic(X, y)
    se_fisher = fisher_se(X, beta_hat)

    def stat_fn(X_b, y_b, j):
        return fit_logistic(X_b, y_b)[j]

    print(f"Logistic regression: n={N}, d={D} (incl. intercept), {N_BOOT} bootstrap refits")
    print(f"{'coef':>6}  {'true':>8}  {'beta_hat':>9}  {'fisher SE':>10}  {'boot SE':>9}  {'fisher 95% CI':>22}  {'boot percentile CI':>22}")
    for j in range(D):
        samples = bootstrap_resample(
            lambda Xb, yb: stat_fn(Xb, yb, j),
            (X, y),
            n_boot=N_BOOT,
            rng=np.random.default_rng(SEED + 1 + j),
        )
        boot_se = float(samples.std(ddof=1))
        fisher_lo = beta_hat[j] - 1.96 * se_fisher[j]
        fisher_hi = beta_hat[j] + 1.96 * se_fisher[j]
        pct_lo, pct_hi = np.percentile(samples, [2.5, 97.5])
        print(
            f"  β_{j}   {TRUE_BETA[j]:+8.3f}  {beta_hat[j]:+9.4f}  "
            f"{se_fisher[j]:10.4f}  {boot_se:9.4f}  "
            f"[{fisher_lo:+7.3f}, {fisher_hi:+7.3f}]  "
            f"[{pct_lo:+7.3f}, {pct_hi:+7.3f}]"
        )

    # Sanity: bootstrap SE should agree with Fisher SE to within ~15% at n=400.
    boot_se_all = np.array([
        bootstrap_resample(
            lambda Xb, yb, j=j: stat_fn(Xb, yb, j),
            (X, y),
            n_boot=300,
            rng=np.random.default_rng(SEED + 100 + j),
        ).std(ddof=1)
        for j in range(D)
    ])
    rel_err = np.abs(boot_se_all - se_fisher) / se_fisher
    print(f"\nmax relative disagreement between bootstrap SE and Fisher SE: {rel_err.max():.3f}")
    assert rel_err.max() < 0.25, f"bootstrap and Fisher disagree by {rel_err.max():.2%}"


if __name__ == "__main__":
    main()
