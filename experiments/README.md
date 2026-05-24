# experiments

Self-contained Kaggle entries and side projects. Larger than
`implementations/` — pipeline-shaped rather than algorithm-shaped, three
to six files per project, real numbers in each README.

| Project | Result |
|---|---|
| [kaggle-f1-pit-stops](kaggle-f1-pit-stops/) | F1 pit-stop multi-anchor blend. Public LB **0.95453**. |
| [tlc-distributed](tlc-distributed/) | 55M NYC taxi trips, three engines on the same weekly aggregate: **DuckDB 2.1s / 2.81 GB** beats Polars (3.0 s / 10.13 GB) and Dask (25.5 s / 13.11 GB). XGBoost tip-percentage RMSE **8.77%** on time-ordered 2023 Nov-Dec holdout. |
| [ais-wti-signal](ais-wti-signal/) | NOAA AIS -> Houston Ship Channel tanker dwell -> WTI. 24 weekly observations, H2 2023. Best feature p90 dwell hours: Pearson **-0.217** (p=0.31), Spearman **-0.240** (p=0.26). Signs right, sample too small for significance. |
