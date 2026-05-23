# F1 Pit Stops

Kaggle: [Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5).
Binary classification — will a car pit on the next lap? Metric is ROC-AUC.

## Scores

| Submission | Public LB |
|---|---|
| CatBoost only | 0.94738 |
| Bagged CatBoost (3 seeds) + OOF target encoding | 0.94776 |
| + sklearn MLP + isotonic calibration | 0.94773 |
| 4-model LR stacker including the ResNet | 0.94943 OOF |
| Stacker blended with a public anchor, w=0.05 | 0.95452 |
| Stacker blended with a 4-anchor logit-rank cascade, w=0.05 | **0.95453** |

The 0.95+ leaderboard is built on a small set of shared submissions; with
only the competition data, our ceiling sits around 0.949. The blend script
takes any external submission as an anchor and mixes our stacker into it.

## Pipeline

- `StratifiedKFold × 5`. Train and test share all 104 races (row-level
  split), so stratified folds mirror the LB.
- Feature engineering over `train + test` sorted by
  `(Race, Year, Driver, LapNumber)`:
  - Lag/lead 1 and 2 of `TyreLife`, `LapTime`, `Stint`, `Compound`, `PitStop`
  - Stint aggregates and progress
  - Compound-normalised `TyreLife` (rebuilt from per-compound quantiles
    since `Normalized_TyreLife` was removed by the host)
  - Field-wide pit rate, same-compound counts at each lap
  - Cumulative driver pit count
  - Oracle: when the next observed lap is exactly N+1, `PitStop_next`
    is the target — exposed as a feature with NaN elsewhere
  - 6 Bayesian-smoothed OOF target encodings
- Models: LightGBM, CatBoost (3 seeds, bagged via logit-rank), XGBoost,
  sklearn MLP, PyTorch residual MLP (3 blocks × 192 hidden, dropout 0.25,
  BCE loss, AUC early stopping, MPS).
- Diagnostics: inter-model OOF correlation (Pearson + Spearman), per-race
  AUC, permutation importance, per-year AUC + calibration, logistic
  stacker with coefficient inspection.
- Final blend: logit-rank of the LR stacker with an optional external
  anchor; isotonic calibration of the chosen prediction.

## Files

| | |
|---|---|
| `f1_pit_stops.ipynb` | EDA, FE, modelling, diagnostics |
| `features.py` | Feature engineering + OOF target encoding |
| `train_resnet.py` | PyTorch residual MLP, saves `oof/resnet.npz` |
| `blend.py` | LR stacker over cached OOFs; optional external anchor |
| `data/` | `train.csv`, `test.csv`, `sample_submission.csv` (gitignored) |
| `oof/` | cached per-model OOF + test predictions (gitignored) |
| `external/` | downloaded third-party submissions (gitignored) |

## Running

```bash
kaggle competitions download -c playground-series-s6e5 -p data/
unzip data/playground-series-s6e5.zip -d data/

jupyter nbconvert --to notebook --execute f1_pit_stops.ipynb
python train_resnet.py

# stacker only
python blend.py

# anchored to a downloaded external submission
kaggle kernels output anthonytherrien/predicting-f1-pit-stops-nn-residual-network \
    -p external/anthony
python blend.py --anchor external/anthony/submission.csv --weight 0.05
```
