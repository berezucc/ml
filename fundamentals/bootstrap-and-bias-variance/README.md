# Bootstrap CIs and bias-variance decomposition

Two foundational statistical ideas that *almost every ML number you produce*
should be wearing:

1. **Bootstrap confidence intervals** — turn a single point estimate into a
   range, with no assumption that the sampling distribution is Gaussian.
2. **Bias-variance decomposition** — explain *why* a model is wrong by
   splitting expected test error into bias², variance, and irreducible noise.

All three scripts are pure NumPy + SciPy, runnable in <30 s.

## Files

| File | What it does | Key result |
|---|---|---|
| `bootstrap.py` | Percentile, basic, and BCa CIs + a permutation test. | 95% CI for mean(Exp(1)) at n=30 achieves 91% empirical coverage across 200 trials. |
| `logistic_se.py` | Logistic regression via Newton-IRLS, then bootstrap coefficient SEs vs closed-form Fisher SEs. | At n=400, the two SE estimates agree to within **17%**, and both CIs cover all four true β's. |
| `bias_variance.py` | Polynomial regression on `sin(2πx) + N(0, 0.3²)`. Estimates bias² and variance pointwise across 300 independent training sets. | Classic U-curve; optimal degree 3, MSE 0.106 vs irreducible floor 0.090. |

## Bootstrap math

Given data `X = (X_1, ..., X_n)` and a statistic `θ̂ = T(X)`, the
non-parametric bootstrap estimates the sampling distribution of `θ̂` by:

1. Resample `X*_1, ..., X*_n` from `X` **with replacement**.
2. Compute `θ̂* = T(X*)`.
3. Repeat `B` times → empirical distribution of `θ̂*`.

Three ways to turn that empirical distribution into a 1-α CI:

    Percentile:   [ θ̂*_{α/2},                    θ̂*_{1-α/2} ]
    Basic:        [ 2θ̂ - θ̂*_{1-α/2},             2θ̂ - θ̂*_{α/2} ]
    BCa:          percentile of (α_lo, α_hi) where the α's are
                  shifted by z_0 (bias-correction, from how often θ̂* < θ̂)
                  and a (acceleration, from jackknife third moment)

When the bootstrap distribution is symmetric, **all three agree**. They
diverge for skewed statistics — common when `θ̂` is near a boundary (AUC
close to 1.0, classification accuracy near 100%, a variance estimate). **BCa
is the only one of the three that's transformation-invariant**, which is why
it's the default in most stats libraries.

## Fisher SE for logistic regression

For logistic regression with MLE `β̂`, the asymptotic covariance is

    Cov(β̂) ≈ (X^T W X)^{-1},     W = diag(p_i (1 - p_i)),  p_i = σ(x_i^T β̂)

so `SE(β̂_j) = sqrt(diag of that)`. This is a *Gaussian approximation* that
holds for large `n` and well-behaved data. The bootstrap is non-parametric:
no Gaussian assumption, no large-n requirement (just a representative
sample). At `n=400` with no separation issues, the two agree closely — and
when they don't, **trust the bootstrap**.

## Bias-variance decomposition

For a regression model `f̂` trained on a random dataset `D`, evaluated at a
fixed `x`, expected test error decomposes as:

    E_D[(y - f̂_D(x))²] = (E_D[f̂_D(x)] - f(x))²    bias²
                       + Var_D[f̂_D(x)]            variance
                       + σ²                       irreducible noise

The decomposition is exact for squared loss. It's *the* reason "more
flexible model" doesn't strictly imply "better." The script estimates each
term empirically: draw `N_DATASETS=300` independent training sets, fit a
polynomial of each degree, average predictions to get the mean function,
then compute bias² and variance pointwise on a grid.

Result (from `python bias_variance.py`):

| deg | bias² | variance | MSE |
|---:|---:|---:|---:|
| 1 | 0.154 | 0.021 | 0.265 |
| 2 | 0.147 | 0.037 | 0.274 |
| 3 | **0.003** | **0.013** | **0.106** ← min |
| 5 | 0.000 | 0.021 | 0.111 |
| 9 | 0.001 | 0.166 | 0.256 |
| 15 | 0.000 | **0.851** | 0.941 |

Bias dies by degree 3 (because `sin(2πx)` on `[0,1]` is well-approximated
by a cubic). After that, the only thing model flexibility buys you is more
variance, and MSE climbs back up.

## Why this matters for the rest of the repo

Every existing test-accuracy and benchmark number could be wearing a `±`.
A few one-liners to drop into other projects:

```python
from bootstrap import bootstrap_ci

def acc(yh, y): return (yh == y).mean()
out = bootstrap_ci(acc, (y_pred, y_test))
print(f"accuracy: {out['theta_hat']:.4f}  95% CI {out['bca']}")
```

For binary classifiers, `roc_auc_score` is the canonical case where BCa
beats percentile (AUC is bounded above by 1 → skewed near top performers).

## Run

    pip install numpy scipy
    python bootstrap.py       # coverage experiment, asserts ≥ 85%
    python logistic_se.py     # bootstrap vs Fisher SE comparison
    python bias_variance.py   # bias-variance U-curve

## Pitfalls

- **The bootstrap fails for max/min and extreme quantiles.** It assumes the
  sample contains a representative tail; for the maximum, the bootstrap
  distribution has a point mass at the observed max, which is wrong.
- **Don't bootstrap dependent data with i.i.d. resampling.** For time
  series, you need block bootstrap. For grouped data, resample groups
  (cluster bootstrap), not rows.
- **BCa's jackknife is O(n)** — slow for large datasets. Subsample or use
  influence-function approximations for production use.
- **Bias-variance for non-squared losses is *not* this clean.** For
  classification 0-1 loss, decompositions exist (Domingos 2000, Kohavi &
  Wolpert 1996) but are messier. The intuition still holds.

## References

- Efron & Tibshirani, *An Introduction to the Bootstrap* (1993) — the textbook.
- Efron, [*Better Bootstrap Confidence Intervals*](https://www.jstor.org/stable/2289144) (1987) — the BCa paper.
- Hastie, Tibshirani, Friedman, *Elements of Statistical Learning*, §7.3 (bias-variance) and §8.2 (bootstrap).
- Domingos, [*A Unified Bias-Variance Decomposition*](https://homes.cs.washington.edu/~pedrod/bvd.pdf) (2000) — generalizes to 0-1 and other losses.
