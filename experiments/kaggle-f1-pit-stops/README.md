# Predicting F1 Pit Stops

Kaggle competition: [Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5)

**Task**: binary classification — will an F1 car pit on the next lap?
**Metric**: ROC-AUC

## Results

| Version | Public LB | Approach |
|---|---|---|
| v3 | 0.94738 | CatBoost on engineered features |
| v4 | 0.94776 | Bagged CB (3 seeds) + OOF target encoding |
| v5 | 0.94773 | + sklearn MLP + LR stacker + isotonic calibration |
| v7 | 0.94924 (OOF) | + PyTorch residual MLP (B2). Best **own-model** OOF AUC |
| v6.1 | **0.95452** | Logit-rank blend of v5 CB-bag with Anthony Therrien's public NN submission (Path A) |
| **v7.1** | **0.95452** | Same blend but using v7's 4-model stacker — ties v6.1 |

## Pipeline overview

- **Validation**: `StratifiedKFold × 5`. Train and test share all 104 races (row-level split), so stratified folds mirror the LB. CV/LB gap < 0.001.
- **EDA**: target rate by compound, pit-hazard curves per compound, `Compound × TyreLife` heatmap, race-track avatars (pit risk painted onto stylized circuit shapes via `RaceProgress`), stint-length distributions, per-year regime check (2023 has 0.96% pit rate vs ~28%).
- **Feature engineering** on combined `train + test`, sorted by `(Race, Year, Driver, LapNumber)`:
  - Lag/lead 1 and 2 of `TyreLife`, `LapTime`, `Stint`, `Compound`, `PitStop`
  - Stint-level aggregates and progress within stint
  - Compound-normalised `TyreLife` (rebuilds the removed `Normalized_TyreLife`)
  - Field-wide pit rate, same-compound counts at this lap
  - Cumulative pit count for this driver in this race
  - **Oracle feature**: when next observed lap is exactly N+1, `PitStop_next` perfectly determines the target
  - 6 OOF target-encoded features: `Driver`, `Race`, `Compound×Stint`, `Driver×Compound`, `Race×Compound`, `Driver×Race`
- **Models**: LightGBM, **CatBoost × 3 seeds (bagged)**, XGBoost, sklearn MLP, **PyTorch residual MLP** (3 residual blocks × 192 hidden, dropout 0.25, BCE loss, AUC early-stopping, MPS-accelerated)
- **Diagnostics**: inter-model OOF correlation (Pearson + Spearman), per-race AUC, permutation importance, per-year AUC + calibration, logistic stacker with coefficient inspection
- **Blend**: logit-rank (AUC-optimal — stretches tails vs linear rank) and LR stacker, isotonic-calibrated final.

### Breaking 0.95 (v6/v7 — both paths)

| Path | Score | What it is |
|---|---|---|
| **Path A** (v6.1) | LB 0.95452 | Logit-rank blend of our CB-bag with [Anthony Therrien's NN-Residual public submission](https://www.kaggle.com/code/anthonytherrien/predicting-f1-pit-stops-nn-residual-network) (his LB 0.95453). Honest note: his "NN" is actually 97% sub1 + 3% sub2 from a private "vault" dataset; the NN weight in his blend is `1e-7`. The 0.95+ public-LB ceiling without leveraging shared submissions is around our own-model 0.9492. |
| **Path B** (v7) | OOF 0.94924 | Pure-ours: trained a real PyTorch residual MLP on our 104 engineered features (OOF 0.93389, Spearman 0.93 with CB-bag — genuine architectural diversity). Stacker including it lifted OOF +0.0001 over CB alone. v7.1 blends this stacker with Anthony at w=0.05, ties v6.1 at 0.95452. |

## Files

- `f1_pit_stops.ipynb` — full v5 pipeline (EDA + FE + 4-model ensemble + diagnostics)
- `train_resnet.py` — Path B: standalone PyTorch residual MLP trainer (saves `oof/resnet.npz`)
- `v6_blend_path_a.py` — Path A blend (Anthony + our CB-bag at w=0.05)
- `v7_blend_path_b.py` — Path B blend (Anthony + our 4-model stacker incl. ResNet at w=0.05)
- `data/` — place `train.csv`, `test.csv`, `sample_submission.csv` here
- `oof/` — cached OOF + test predictions per model (instant re-blending)
- `external/anthony/` — Anthony's downloaded NN submission (gitignored)

## Running

```bash
# Data
kaggle competitions download -c playground-series-s6e5 -p data/ && \
  unzip data/playground-series-s6e5.zip -d data/

# Main pipeline (saves oof/{lgb,cb_bag,xgb,...}.npz, submission.csv at LB 0.94773)
jupyter nbconvert --to notebook --execute f1_pit_stops.ipynb

# Path B: train the residual MLP (~5 min on Apple MPS)
python train_resnet.py

# Path A or B blend with Anthony's public submission
kaggle kernels output anthonytherrien/predicting-f1-pit-stops-nn-residual-network \
    -p ./external/anthony
python v6_blend_path_a.py   # OR v7_blend_path_b.py — both score LB 0.95452
```
