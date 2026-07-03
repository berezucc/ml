# TabFM (zero-shot) vs CatBoost — F1 pit-stop prediction

Evaluates Google's [TabFM](https://github.com/google-research/tabfm), a 1.6B-parameter
zero-shot tabular foundation model
([blog post](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)),
against CatBoost on the [Kaggle Playground S6E5](https://www.kaggle.com/competitions/playground-series-s6e5)
pit-stop data. The tuned CatBoost + feature-engineering baseline lives in
[`../kaggle-f1-pit-stops`](../kaggle-f1-pit-stops); this experiment reuses its data,
feature pipeline and CV protocol so the numbers are directly comparable.

The S6E5 data was synthetically generated in May 2026, after TabFM's release, so the
model cannot have seen it in pretraining.

## Layout

- `experiment.py` — data loading, TabFM/CatBoost runners; every run cached to `artifacts/runs/<key>.json`
- `run_all.py` — precomputes all runs with progress logging (`artifacts/run.log`)
- `tabfm_vs_catboost.ipynb` — renders tables and charts from the cached runs (seconds on a warm cache)

```bash
nohup caffeinate -i .venv/bin/python run_all.py > artifacts/run.log 2>&1 &
```

## Experiments (`tabfm_vs_catboost.ipynb`)

- **A. Model x features grid** — TabFM zero-shot vs CatBoost, on raw competition columns
  vs the ~100-feature engineered set, all scored on one shared fold-0 eval sample.
  CatBoost appears both row-budget-matched to TabFM's context and trained on the full fold.
- **B. Context-size sweep** — AUC and wall-clock vs rows available (500 → 10k),
  TabFM in-context vs CatBoost trained on the same rows.

## Setup

```bash
uv venv --python 3.12 .venv
git clone https://github.com/google-research/tabfm vendor/tabfm
uv pip install -e "./vendor/tabfm[pytorch]" safetensors catboost pyarrow matplotlib ipykernel
```

Pretrained v1.0.0 weights download from Hugging Face Hub on first model load.
Data is expected at `../kaggle-f1-pit-stops/data/{train,test}.csv` (from the competition page).

## Notes

- TabFM's PyTorch backend runs on Apple Silicon via MPS (`model.to("mps")`), ~17x faster
  than CPU here. Inference is still minutes per run; timing is reported alongside AUC.
- `n_estimators=8` (default 32) for speed; on this data the AUC difference was negligible.
- Results are cached in `artifacts/results_*.csv`; set `RERUN = True` in the notebook to recompute.
