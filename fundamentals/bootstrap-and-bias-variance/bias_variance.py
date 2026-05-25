"""Bias-variance decomposition for polynomial regression on f(x) = sin(2πx) + noise.

For each polynomial degree d, we draw many independent training sets, fit a
model on each, then estimate at a grid of test points:
    bias²(x)   = (E_D[f̂_d(x)] - f(x))²
    variance(x) = Var_D[f̂_d(x)]
    test MSE   = bias² + variance + σ²    (the decomposition)

The plot you'd draw shows the classic U-curve: bias falls as d grows, variance
explodes, MSE bottoms out at a sweet spot.
"""
from __future__ import annotations

import numpy as np

SEED = 0
N_TRAIN = 30          # size of each bootstrap training set
N_DATASETS = 300      # number of independent training sets
SIGMA = 0.3           # noise stddev
DEGREES = list(range(1, 16))


def f_true(x):
    return np.sin(2 * np.pi * x)


def make_dataset(rng, n=N_TRAIN, sigma=SIGMA):
    x = rng.uniform(0.0, 1.0, size=n)
    y = f_true(x) + rng.normal(0.0, sigma, size=n)
    return x, y


def fit_poly(x, y, degree):
    # Plain least squares via numpy. Ridge of 1e-8 stabilizes high-degree fits.
    X = np.vander(x, degree + 1, increasing=True)
    XtX = X.T @ X + 1e-8 * np.eye(degree + 1)
    return np.linalg.solve(XtX, X.T @ y)


def predict_poly(coef, x):
    return np.polynomial.polynomial.polyval(x, coef)


def decomposition():
    rng = np.random.default_rng(SEED)
    x_test = np.linspace(0.05, 0.95, 100)
    y_true_test = f_true(x_test)

    results = []
    for d in DEGREES:
        # For each dataset, fit a model; store predictions on the test grid.
        preds = np.empty((N_DATASETS, len(x_test)))
        for k in range(N_DATASETS):
            x_tr, y_tr = make_dataset(rng)
            coef = fit_poly(x_tr, y_tr, d)
            preds[k] = predict_poly(coef, x_test)
        mean_pred = preds.mean(axis=0)
        # Pointwise bias² and variance, then average over the test grid.
        bias_sq = float(((mean_pred - y_true_test) ** 2).mean())
        var = float(preds.var(axis=0).mean())
        mse_expected = bias_sq + var + SIGMA ** 2
        results.append((d, bias_sq, var, mse_expected))
    return results


def main():
    print(f"f(x) = sin(2πx) + N(0, {SIGMA}²),  n_train = {N_TRAIN},  n_datasets = {N_DATASETS}\n")
    print(f"{'deg':>4}  {'bias²':>10}  {'variance':>10}  {'noise':>8}  {'MSE':>10}")
    print(f"{'':>4}  {'':>10}  {'':>10}  {SIGMA**2:>8.4f}  {'':>10}")
    print("-" * 50)
    results = decomposition()
    for d, b, v, m in results:
        marker = ""
        if (d, b, v, m) == min(results, key=lambda r: r[3]):
            marker = "  ← min"
        print(f"{d:>4}  {b:>10.4f}  {v:>10.4f}  {'':>8}  {m:>10.4f}{marker}")

    # Sanity checks for the classic U-curve.
    biases = [b for _, b, _, _ in results]
    variances = [v for _, _, v, _ in results]
    # Bias should fall sharply early (deg 1 vs deg 5).
    assert biases[0] > biases[4], f"bias not monotonic enough early: {biases[:5]}"
    # Variance should explode for high degree (deg 1 vs deg 15).
    assert variances[-1] > 5 * variances[0], (
        f"variance didn't blow up at high deg: {variances[0]:.4f} -> {variances[-1]:.4f}"
    )
    best_deg, *_ = min(results, key=lambda r: r[3])
    print(f"\noptimal degree: {best_deg}  (irreducible noise floor = {SIGMA**2:.4f})")


if __name__ == "__main__":
    main()
