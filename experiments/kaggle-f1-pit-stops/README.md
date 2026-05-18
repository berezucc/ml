# Predicting F1 Pit Stops

Kaggle competition: [Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5)

**Task**: binary classification — will an F1 car pit on the next lap?
**Metric**: ROC-AUC
**Result**: 0.94738 public LB (CatBoost alone)

## Approach

- **Validation**: `StratifiedKFold × 5`. Train and test share all 104 races (row-level split), so stratified folds mirror the LB. CV/LB gap < 0.001.
- **EDA**: target rate by compound, pit-hazard curves per compound, `Compound × TyreLife` heatmap, race-track avatars (pit risk painted onto stylized circuit shapes using `RaceProgress`), stint-length distributions.
- **Feature engineering** on `train + test` combined, sorted by `(Race, Year, Driver, LapNumber)`:
  - Lag/lead 1 and 2 of `TyreLife`, `LapTime`, `Stint`, `Compound`, `PitStop`
  - Stint-level aggregates and progress within stint
  - Compound-normalised `TyreLife` (rebuilds the removed `Normalized_TyreLife`)
  - Field-wide pit rate, same-compound counts at this lap
  - Cumulative pit count for this driver in this race
  - **Oracle feature**: when the next observed lap is exactly N+1, `PitStop_next` perfectly determines the target
- **Models**: LightGBM + CatBoost + XGBoost, 5-fold OOF cached to `oof/`
- **Diagnostics**: inter-model OOF correlation, per-race AUC (find weakest slices), permutation importance on a 40k val sample
- **Blend**: logit-rank (AUC-optimal: stretches tails vs linear rank). Final submission picks the best of {single, linear rank, logit-rank} by OOF AUC — CatBoost alone won.

## Files

- `f1_pit_stops.ipynb` — full pipeline
- `data/` — place `train.csv`, `test.csv`, `sample_submission.csv` here
- `oof/` — cached OOF + test predictions per model (for instant re-blending)

## Running

```bash
kaggle competitions download -c playground-series-s6e5 -p data/ && \
  unzip data/playground-series-s6e5.zip -d data/
jupyter nbconvert --to notebook --execute f1_pit_stops.ipynb
```
