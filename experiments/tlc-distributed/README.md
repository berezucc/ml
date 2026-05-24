# NYC TLC distributed pipeline + tip-pct model

55 M NYC taxi trips (yellow + green + FHV, 2023), three data engines on
the same weekly-aggregation query, one XGBoost tip-percentage model on
the result.

## Result: weekly (PU zone, DO zone, weekday) aggregate over 55 M trips

Same query, three engines, three runs each. Median wall + max peak RSS:

| engine  | wall-clock (median of 3) | peak RSS | output |
|---------|-------------------------:|---------:|-------:|
| DuckDB  |        2.1 s             | 2.81 GB  | 27.0 MB|
| Polars  |        3.0 s             |10.13 GB  | 30.2 MB|
| Dask    |       25.5 s             |13.11 GB  | 23.7 MB|

3.90 M output rows / 42.54 M aggregated trips. All three engines agree on
row count and aggregate sums within float epsilon (validated in `bench.py`).

DuckDB wins both axes: 1.4x faster than Polars and 3.6x less memory; 12x
faster than Dask and 4.7x less memory. Dask carries scheduler overhead
that doesn't pay off on a single node — its value is fitting workloads
that don't fit on one machine, which this one does.

## Result: tip-percentage model (yellow + green)

**RMSE 0.0877 (8.77%)** on a time-ordered holdout (train Jan-Oct 2023,
test Nov-Dec). 100 s training on all M2 Max cores, 31.3 M train rows,
6.6 M test rows, 200 boost rounds, early stop 20.

Top-5 feature importance (gain):

| feature        |   gain |
|----------------|-------:|
| payment_type   |  494.9 |
| trip_minutes   |   18.0 |
| trip_distance  |    6.2 |
| pickup_hour    |    3.2 |
| passenger_count|    2.7 |

`payment_type` dominates by 27x — cash trips have `tip_amount` recorded as
zero, card trips cluster at 18-25%. The model is mostly learning the
payment channel, not the trip. RMSE plateaus at ~8.8% after ~40 rounds.

## Run

```bash
pip install pyarrow polars dask[dataframe] duckdb xgboost psutil requests
python ingest.py          # ~25 min, ~1 GB download, 36 hive-partitions
python bench.py           # ~3 min, 3 engines × 3 runs each
python train.py           # ~2 min on M2 Max
```

Scope this run: 2023 only. Full 2019-2024 (~500 M trips) by changing
`YEAR_START` / `YEAR_END` in `ingest.py`.

## Note on the stack

The spec called for `xgboost-ray` for distributed training. `xgboost-ray`
0.1.20 crashes on XGBoost 2.x with
`TypeError: getaddrinfo() argument 1 must be string or None` inside
`_start_rabit_tracker` — a known compat bug with newer XGBoost. The
documented fallback is plain XGBoost with `nthread=-1`, which on a
single M2 Max gets the same throughput as a 4-actor Ray cluster would,
just without the Ray scheduling overhead.

## References

- NYC TLC trip-record data dictionary and parquet flat files
  (`d37ci6vzurychx.cloudfront.net/trip-data`)
- Polars 1.x lazy + streaming engine docs
- Dask DataFrame and DuckDB docs (the out-of-core sections)
- XGBoost `hist` tree method docs
