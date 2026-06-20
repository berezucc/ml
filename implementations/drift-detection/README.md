# Drift detection from scratch

Six ways to ask "does today's data still look like training data?", each
implemented in NumPy and checked against SciPy, then applied to a logistic
regression on the Wisconsin breast-cancer set with covariate shift injected into
three features. Notebook: [`drift_detection.ipynb`](drift_detection.ipynb).

No `evidently` / `alibi-detect` - the point is the maths.

## What's in it

| Metric | Sees | Reference checked against |
|---|---|---|
| Summary stats + **Welch's t-test** | mean shift only | `scipy.stats.ttest_ind(equal_var=False)` |
| **PSI** (population stability index) | binned shift, symmetric | thresholds 0.10 / 0.25 |
| **KS** two-sample | worst CDF gap (weak in tails) | `scipy.stats.ks_2samp` |
| **Wasserstein** (1-D earth mover's) | total mass moved, in feature units | `scipy.stats.wasserstein_distance` |
| **KL** divergence | binned, asymmetric, unbounded | `scipy.stats.entropy` |
| **Jensen-Shannon** | binned, symmetric, bounded by `ln 2` | `scipy.spatial.distance.jensenshannon` |

Every from-scratch metric matches SciPy to `|diff| < 1e-9` (most to machine
precision); the notebook asserts this inline.

## Result

Production = the reference window with a graded covariate shift injected into
three features and everything else left identical, so untouched features read
exactly zero and only the injected drift shows up. Held-out AUC moves
`0.9965 -> 0.9966` - the model still ranks cases while the inputs have clearly
moved, which is the whole reason to watch inputs rather than wait for labels.

| feature | PSI | KS | Wasserstein | JS | welch p | flag |
|---|---:|---:|---:|---:|---:|---|
| mean radius | 1.0597 | 0.444 | 3.05 | 0.128 | ~0 | **major** |
| mean texture | 0.2445 | 0.199 | 2.04 | 0.036 | ~0 | moderate |
| worst area | 0.1771 | 0.164 | 79.21 | 0.029 | 0.050 | moderate |
| mean smoothness | 0.0000 | 0.000 | 0.00 | 0.000 | 1.000 | stable |
| mean symmetry | 0.0000 | 0.000 | 0.00 | 0.000 | 1.000 | stable |
| predicted prob | 0.0219 | 0.059 | 0.02 | 0.006 | 0.588 | stable |

Things the table teaches that one metric alone wouldn't:

- **Wasserstein isn't scale-free.** `worst area` reads 79.21 not because it
  drifted most but because it lives on a large scale - useless for ranking
  across features, great for "how many units moved" on one.
- **The t-test is on the fence** for `worst area` (p = 0.05) while PSI already
  flags it - the mean barely moved but the distribution shape did.
- **PSI is symmetric only for fixed bins.** Anchoring edges to the reference
  (the right thing to do) makes `PSI(ref,prod)=1.0597` differ from
  `PSI(prod,ref)=1.0073`; the per-bin term itself is symmetric (shown in the
  notebook).

## Run

    pip install numpy scipy scikit-learn pandas matplotlib
    jupyter nbconvert --to notebook --execute --inplace drift_detection.ipynb

## Pitfalls (learned writing this)

- **Binning controls PSI/KL/JS.** Fix the scheme (10 quantile bins), take edges
  from the reference only, keep it constant over time - otherwise the trend is
  an artefact of the bins.
- **Empty bins send the logs to `inf`.** Floor proportions with an `eps`; that
  `eps` quietly caps each bin's contribution.
- **p-values scale with `n`.** Big windows make KS / t-test p ~ 0 for drift too
  small to care about. Threshold on effect size (PSI, KS statistic,
  Wasserstein), not significance.
- **Covariate drift is not concept drift.** These are an early-warning layer,
  not a substitute for backtesting against labels once they arrive.

## References

- PSI thresholds - Yurdakul, *Statistical Properties of the Population Stability
  Index* (WMU PhD thesis, 2018), <https://scholarworks.wmich.edu/dissertations/3208/>.
- KS two-sample - Hodges, *The significance probability of the Smirnov two-sample test* (1958).
- Wasserstein two-sample - Ramdas, Garcia, Cuturi (2017), <https://arxiv.org/abs/1509.02237>.
- KL - Kullback & Leibler, *On Information and Sufficiency* (1951).
- Jensen-Shannon - Lin, *Divergence Measures Based on the Shannon Entropy*, IEEE TIT (1991), <https://ieeexplore.ieee.org/document/61115>.
- Drift taxonomy - Gama et al., *A Survey on Concept Drift Adaptation*, ACM CSUR (2014), <https://dl.acm.org/doi/10.1145/2523813>.
