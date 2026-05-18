# Predicting F1 Pit Stops

Kaggle competition: [Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5)

**Task**: binary classification — will an F1 car pit on the next lap?
**Metric**: ROC-AUC
**Result**: 0.94689 public LB

## Approach

- **Validation**: `StratifiedKFold × 5`. Train and test share all 104 races (row-level split), so stratified folds mirror the LB. CV/LB gap was 0.00015.
- **Feature engineering** on `train + test` combined, sorted by `(Race, Year, Driver, LapNumber)`:
  - Lag/lead 1 and 2 of `TyreLife`, `LapTime`, `Stint`, `Compound`, `PitStop`
  - Stint-level aggregates and progress within stint
  - Compound-normalised `TyreLife` (rebuilds the removed `Normalized_TyreLife`)
  - Field-wide pit rate, same-compound counts at this lap
  - Cumulative pit count for this driver in this race
  - **Oracle feature**: when the next observed lap is exactly N+1, `PitStop_next` perfectly determines the target
- **Models**: LightGBM + CatBoost, weighted-rank ensemble by OOF AUC²

## Files

- `f1_pit_stops.ipynb` — full pipeline
- `data/` — place `train.csv`, `test.csv`, `sample_submission.csv` here

## Running

```bash
kaggle competitions download -c playground-series-s6e5 -p data/ && \
  unzip data/playground-series-s6e5.zip -d data/
jupyter nbconvert --to notebook --execute f1_pit_stops.ipynb
```
