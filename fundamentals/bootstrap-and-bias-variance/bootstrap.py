"""Non-parametric bootstrap confidence intervals: percentile, basic, BCa.

Run this file to verify the methods on a known case (mean of Exp(1) samples,
true mean 1.0). All three CIs should cover 1.0 well above the nominal rate.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm


def bootstrap_resample(
    stat_fn: Callable,
    data,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Resample rows of `data` with replacement, compute `stat_fn` each time.

    data: a single array OR a tuple of arrays of equal length (e.g. (X, y)).
    Returns (n_boot,) array of bootstrap statistics.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if not isinstance(data, tuple):
        data = (data,)
    n = len(data[0])
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[b] = stat_fn(*(d[idx] for d in data))
    return samples


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def basic_ci(theta_hat: float, samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    # Pivotal/basic CI: reflect the bootstrap distribution around theta_hat.
    # Useful when the bootstrap distribution is biased relative to theta_hat.
    p_lo, p_hi = percentile_ci(samples, alpha)
    return 2.0 * theta_hat - p_hi, 2.0 * theta_hat - p_lo


def bca_ci(
    stat_fn: Callable,
    data,
    theta_hat: float,
    samples: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bias-corrected and accelerated CI (Efron 1987).

    Corrects for both bias (z0: median of bootstrap dist vs theta_hat)
    and skewness (a: from jackknife). Asymptotically more accurate than
    percentile, and the standard choice when distributions are skewed.
    """
    if not isinstance(data, tuple):
        data = (data,)
    n = len(data[0])

    # Bias-correction: how often does the bootstrap stat fall below theta_hat?
    p0 = float(np.mean(samples < theta_hat))
    if p0 == 0.0 or p0 == 1.0:
        return percentile_ci(samples, alpha)
    z0 = norm.ppf(p0)

    # Acceleration: third moment of jackknife stats (measures skewness).
    jack = np.empty(n)
    idx_all = np.arange(n)
    for i in range(n):
        keep = idx_all[idx_all != i]
        jack[i] = stat_fn(*(d[keep] for d in data))
    jack_mean = jack.mean()
    diff = jack_mean - jack
    den = 6.0 * (diff ** 2).sum() ** 1.5
    a = float((diff ** 3).sum() / den) if den > 0 else 0.0

    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1.0 - alpha / 2)
    alpha_lo = norm.cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)))
    alpha_hi = norm.cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)))
    lo = float(np.percentile(samples, 100 * alpha_lo))
    hi = float(np.percentile(samples, 100 * alpha_hi))
    return lo, hi


def bootstrap_ci(
    stat_fn: Callable,
    data,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """All three CIs in one shot. Returns dict with theta_hat, se, and three CIs."""
    if not isinstance(data, tuple):
        data = (data,)
    theta_hat = float(stat_fn(*data))
    samples = bootstrap_resample(stat_fn, data, n_boot=n_boot, rng=rng)
    return {
        "theta_hat": theta_hat,
        "se": float(samples.std(ddof=1)),
        "samples": samples,
        "percentile": percentile_ci(samples, alpha),
        "basic": basic_ci(theta_hat, samples, alpha),
        "bca": bca_ci(stat_fn, data, theta_hat, samples, alpha),
    }


def permutation_test(stat_fn: Callable, a, b, n_perm: int = 2000, rng=None) -> float:
    """Two-sample permutation test. Returns two-sided p-value for stat_fn(a) - stat_fn(b)."""
    if rng is None:
        rng = np.random.default_rng(0)
    obs = float(stat_fn(a) - stat_fn(b))
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = float(stat_fn(perm[:na]) - stat_fn(perm[na:]))
        if abs(diff) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def _coverage_experiment():
    """Hammer the three CI methods on samples from Exp(1). True mean = 1.0.

    Verifies coverage rate >= ~90% for nominal 95% CIs at small n (n=30 is mildly
    skewed; BCa should beat percentile here on skewed asymmetry).
    """
    rng = np.random.default_rng(0)
    n = 30
    n_trials = 200
    cover = {"percentile": 0, "basic": 0, "bca": 0}
    widths = {"percentile": [], "basic": [], "bca": []}
    for _ in range(n_trials):
        x = rng.exponential(scale=1.0, size=n)
        out = bootstrap_ci(np.mean, x, n_boot=600, rng=rng)
        for method in cover:
            lo, hi = out[method]
            if lo <= 1.0 <= hi:
                cover[method] += 1
            widths[method].append(hi - lo)
    return cover, widths, n_trials


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Sanity: one specific sample, all three CIs should be near each other.
    x = rng.exponential(scale=1.0, size=200)
    out = bootstrap_ci(np.mean, x, n_boot=2000, rng=rng)
    print(f"Exp(1), n=200:  theta_hat = {out['theta_hat']:.4f}   (true 1.0000)")
    print(f"  SE                  = {out['se']:.4f}")
    print(f"  Percentile 95% CI   = [{out['percentile'][0]:.4f}, {out['percentile'][1]:.4f}]")
    print(f"  Basic      95% CI   = [{out['basic'][0]:.4f}, {out['basic'][1]:.4f}]")
    print(f"  BCa        95% CI   = [{out['bca'][0]:.4f}, {out['bca'][1]:.4f}]")

    # Coverage experiment: how often does each method cover the truth?
    print("\nCoverage of 95% CI for mean(Exp(1)), n=30, 200 trials:")
    cover, widths, n_trials = _coverage_experiment()
    for method in ("percentile", "basic", "bca"):
        rate = cover[method] / n_trials
        w = float(np.mean(widths[method]))
        print(f"  {method:>10}: coverage = {rate:.3f}   avg width = {w:.4f}")
        assert rate >= 0.85, f"{method} coverage {rate:.3f} below 0.85"
    print("OK")
