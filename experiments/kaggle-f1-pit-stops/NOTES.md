# Notes

## Problem

Kaggle Playground Series S6E5 asks for the probability that an F1 car
will pit on the next lap. Binary target `PitNextLap`, metric ROC AUC.
439k training rows, 188k test rows, 16 columns.

## Data audit

A few facts about the data shaped every later choice.

The host removed `Normalized_TyreLife` from the published columns. Raw
`TyreLife` and `Stint` are still there, so a reasonable substitute can
be reconstructed by dividing `TyreLife` by per-compound quantiles.

Train and test share all 104 races. The split is row-level random
within each race, not race-level holdout. Stratified K-fold on the
training set therefore mirrors the public leaderboard. Cross-validated
AUC tracked the LB to within 0.001 throughout.

The `PitStop` column is present for every test row. When the next
observed row for the same driver in the same race is exactly the next
lap number, `PitStop_next` is the target. This is legal because all
test inputs are visible at inference, and it directly determines the
target for roughly two thirds of rows. The feature is exposed as
`oracle_pit_next` (NaN otherwise).

The 2023 season is anomalous. Pit rate sits around 1 percent against
about 28 percent for the other three years. The original dataset has
the same skew. Models pick up `Year` as a regime feature, which lifts
2023 AUC but produces predictions that are badly miscalibrated against
the per-year base rate. Isotonic regression on out-of-fold predictions
restores calibration without affecting AUC.

## Feature engineering

All features are computed on the concatenated train and test frame
sorted by race, year, driver, lap number. This lets neighbour lookups
work uniformly across both sets.

* Shift-1 and shift-2 lags and leads of `TyreLife`, `LapTime`, `Stint`,
  `Compound`, `PitStop`, `Position`, `LapNumber`, and related columns.
* Stint-level aggregates: max tyre life, stint length, progress
  fraction, laps remaining in stint.
* Compound normalisation: `TyreLife` divided by the 95th percentile of
  `TyreLife` for that compound. Reproduces a signal close to the
  removed `Normalized_TyreLife`.
* Race context: mean position, mean lap time, field-wide pit rate,
  fraction of cars on the same compound at this lap.
* Driver progress: cumulative pit count so far, fraction of race
  remaining.
* Oracle features described above.
* Bayesian-smoothed out-of-fold target encoding (alpha=20) for six
  groups: `Driver`, `Race`, `Compound x Stint`, `Driver x Compound`,
  `Race x Compound`, `Driver x Race`.

Final feature set is 104 columns.

## Models

Each model trains 5 stratified folds and caches OOF and test
predictions to `oof/`.

| Model | OOF AUC |
| --- | --- |
| LightGBM | 0.9425 |
| CatBoost (single seed) | 0.9471 |
| Bagged CatBoost (3 seeds, varied depth and learning rate) | 0.9491 |
| XGBoost | 0.9457 |
| sklearn MLP | 0.9321 |
| PyTorch residual MLP | 0.9339 |

Bagged CatBoost is the strongest single model. The +0.0021 gain over a
single seed comes from hyperparameter diversity (depth 7, 8, 9 and
learning rate 0.06, 0.05, 0.04), not just from seed variance.

The neural networks are weaker but have the lowest Spearman correlation
with the GBDTs (around 0.93). They are useful for blend diversity even
though they cannot beat CatBoost on their own.

## Blending

Three strategies were tested.

1. Logit-rank average weighted by AUC squared. Output is mapped back to
   the anchor's value distribution.
2. Logistic regression stacker over logit-transformed OOF ranks.
3. Logit-rank blend with an external anchor submission, weight 0.05.

The stacker beats the best single model by 0.0001. The 4-way logit-rank
average underperforms because the weaker models drag the stronger ones.

Isotonic regression is fit on the chosen OOF predictions and applied to
the chosen test predictions. AUC is preserved. The predicted mean
matches the training base rate of 0.199 to four decimal places.

## Leaderboard progression

| Submission | Public LB | Note |
| --- | --- | --- |
| CatBoost only | 0.9474 | baseline |
| Bagged CatBoost + TE | 0.9478 | best from own model alone |
| + sklearn MLP, isotonic | 0.9477 | flat |
| Stacker including residual MLP | 0.9492 OOF | own-model ceiling |
| Blend with public anchor, w=0.05 | 0.9545 | external help |

The 0.95+ leaderboard is built from a small set of shared submissions
that public blender notebooks pass around. Reproducing one of those
notebooks shows the headline model contributes weight 1e-7 in the
final blend; the score comes from two other submissions read from a
private dataset. With only the published data, the own-model ceiling
is around 0.949.

## Things tried that did not help

Per-fold target encoding matrices. Built a leak-free version that
recomputes TE statistics from each fold's training data only and
applies them to both the training and validation rows of that fold.
LightGBM OOF dropped from 0.94248 (current single-OOF TE) to 0.93870.
Reading: the leak in the current TE pattern was inflating OOF more
than helping generalisation. The TE features on this problem do not
carry real predictive power beyond what CatBoost extracts natively;
the apparent gain from TE in v4 came from the OOF-measurement leak,
not from improved test-time prediction.

## What to try next

Optuna tuning on the bagged CatBoost. The current configuration uses
sensible defaults but was not tuned. A search over depth, learning
rate, l2 leaf reg, bagging temperature, and random strength is the
next obvious move. Expected gain 0.001 to 0.003.

Pseudo-labelling on test rows where the stacker is highly confident
(p above 0.95 or below 0.05). Train an extra CatBoost on the augmented
set. Modest expected gain with some confirmation-bias risk.

Drop the TE features entirely and remeasure. Given the per-fold result,
the existing TE may be net-negative under honest evaluation.

## Files

```
features.py          feature engineering and OOF target encoding
train_resnet.py      PyTorch residual MLP, writes oof/resnet.npz
blend.py             LR stacker, optional external anchor
f1_pit_stops.ipynb   EDA, modelling, diagnostics
README.md            usage
```

Cached out-of-fold and test predictions live in `oof/` for fast
re-blending. The `external/` directory holds third-party submissions
used as anchors. Both are gitignored.
