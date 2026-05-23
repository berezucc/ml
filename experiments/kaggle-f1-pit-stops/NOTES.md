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
The leak in the current TE pattern was inflating OOF more than
helping generalisation. The TE features on this problem do not carry
real predictive power beyond what CatBoost extracts natively; the
apparent gain from TE in v4 came from the OOF-measurement leak, not
from improved test-time prediction.

GBDT stacker (small LightGBM, num_leaves=15) over the four model
OOFs. OOF 0.94897 vs LR stacker 0.94924. LR wins, probably because
LR's implicit linear regularisation is well-suited to combining four
highly correlated GBDT-flavoured signals.

## Diagnostics

Adversarial validation. Train LightGBM to classify train vs test.
With all features: AUC 1.00 (perfect drift), driven entirely by
`te_Race`. Dropping the TE columns: AUC 0.50 (no drift). The TE
features have a train/test distribution mismatch because train uses
per-fold aggregates while test uses a full-train aggregate.

SHAP on a fold-0 CatBoost (1200 iter), 20k validation sample. Top
features by mean |SHAP|: Year (0.86), TyreLife (0.53),
te_Compound_x_Stint (0.39), Race (0.39), LapTime_Delta (0.36),
tyre_life_normalised (0.34). Year dominates because of the 2023
regime split. Oracle ranks 14 with mean |SHAP| 0.09, lower than
expected, but it is only valid for ~67% of rows so the average is
diluted. Plots in `diagnostics_out/`.

## Optuna tuning on CatBoost

30-trial search over depth, learning rate, l2 leaf reg, bagging
temperature, random strength on a single-fold (fold 0) validation.
Total wall time about 2 hours. Best params:

```
iterations: 2500
depth: 10
learning_rate: 0.065
l2_leaf_reg: 3.97
bagging_temperature: 0.156
random_strength: 0.89
```

Best fold-0 AUC 0.94839 vs the existing depth-8 default of about 0.9474.

Trained these params across all 5 folds (`train_tuned_cb.py`,
`oof/cb_tuned.npz`). Per-fold AUCs: 0.94839, 0.94709, 0.94941, 0.94876,
0.94842. Full OOF 0.94841, which is worse than two of the three existing
CB seeds. The single-fold tune overfits to fold 0.

Swapping the tuned CB in for the weakest existing seed (cb_1, OOF
0.94812) gave a 3-bag OOF of 0.94936, up from 0.94914 (+0.0002). The
LR stacker over (lgb, new-bag, xgb, resnet) reaches OOF 0.94943, up
from 0.94924. The anchored submission with Anthony at w=0.05 is
unchanged on the LB at 0.95452, because the blend at that weight is
dominated by the anchor.

## What to try next

Pseudo-labelling on test rows where the stacker is highly confident
(p above 0.95 or below 0.05). Train an extra CatBoost on the augmented
set. Modest expected gain with some confirmation-bias risk.

Multi-fold Optuna instead of single-fold. The single-fold tune overfits.
Five-fold Optuna would take ten times as long but the best params should
generalise better across folds.

Drop the TE features entirely and remeasure. Adversarial validation
shows the TE columns are the only source of train/test distribution
drift (AUC 1.00 with TE, 0.50 without).

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
