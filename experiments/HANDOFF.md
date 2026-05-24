# Experiments: handoff plan

Two projects to add to `experiments/`. The first is an alt-data signal
end-to-end: ingest, clean, feature, backtest, write up an honest result.
The second is a distributed-compute / data-engineering project that
exercises the modern OSS data stack (Polars, Dask, DuckDB, Ray) on a
real public dataset.

Both are scoped to 3-4 days of focused work. Both produce a measurable
number — a signal IC or a perf-comparison table — and a writeup that is
honest about what doesn't work.

## House style

Same conventions as `implementations/HANDOFF.md`:

| Question | Answer |
| --- | --- |
| Python version | 3.11+ |
| Type hints | Sparingly on public functions, never on locals |
| Logging | `print()`, not `logging` |
| Tests | One or two `assert`s in `if __name__ == "__main__"`, no pytest |
| Random seeds | Set explicitly at top of every script |
| Configuration | Hardcoded constants at top of file, `argparse` only if a one-shot script genuinely needs CLI args |
| Dependencies | One line at top of each README, no per-project `requirements.txt` unless install is non-trivial |
| Comments | Only where a reader cannot reverse-engineer the why from the code |
| README results | Concrete numbers. "IC 0.14 on 2-year holdout" not "promising signal" |

These projects can be slightly larger than the `implementations/`
projects — three to six files per project is fine since the work is
pipeline-shaped rather than algorithm-shaped.

---

## Project 1: AIS tanker dwell times -> WTI signal

```
experiments/ais-wti-signal/
  README.md
  data.py         # download + filter AIS, write partitioned parquet
  features.py     # port polygon, tanker dwell times, weekly aggregates
  signal.py       # join WTI prices, compute IC + event study
  data/           # gitignored, raw + processed parquet
```

### Scope

Ingest US Coast Guard AIS records for the Houston Ship Channel for
2022-2023. Filter to crude tankers. For each tanker, identify port
entry/exit events via point-in-polygon over consecutive pings. Aggregate
to weekly arrival count and median dwell hours. Join to WTI front-month
futures, lag features by one week (no lookahead), compute IC vs 5-day
forward WTI returns. Run an event study around weeks where dwell time
is more than two standard deviations from rolling mean.

The target is honest methodology, not a magic Sharpe. A small positive
or negative IC is publishable. Lookahead, survivorship, and selection
issues get called out by name in the writeup.

### Data

- AIS records: NOAA `marinecadastre.gov` "AIS Vessel Traffic Data",
  Zone 15, 2022 + 2023. ~80 GB uncompressed; ~6 GB as filtered parquet.
- Vessel registry: AIS messages include `VesselType`. Filter to `80-89`
  (tankers). For higher precision, join `MMSI` to a downloaded ITU
  registry snapshot.
- Houston Ship Channel polygon: hand-drawn GeoJSON around the
  ~50-mile channel from Galveston Bay to the Turning Basin (one file,
  ~20 vertices).
- WTI front-month: `yfinance` ticker `CL=F`, or FRED series `DCOILWTICO`
  for spot.

### Files

- `data.py` (~80 lines)
  - Loop over months, download zip from marinecadastre, unzip in stream,
    filter to bounding box + tanker types, append to monthly parquet.
  - One `if __name__ == "__main__"` assert: parquet has expected columns
    and at least N rows for a known month.
- `features.py` (~120 lines)
  - Load polygon, sort pings by `(MMSI, BaseDateTime)`, mark in-polygon,
    detect entry/exit transitions, compute dwell hours per visit,
    aggregate weekly.
  - Output: `weekly_features.parquet` with columns `week, n_visits, median_dwell_hours, total_dwell_hours`.
- `signal.py` (~100 lines)
  - Load WTI, compute 5-day forward log returns, downsample to weekly.
  - Lag features by one week. Compute Pearson + Spearman IC. Run event
    study: returns in the 2 weeks after a top/bottom 10% feature week.
  - Print a small markdown table; save to README via copy-paste.
- `README.md` (50-60 lines)

### References

- NOAA `marinecadastre.gov` AIS data dictionary and example notebooks
- Brancaccio, Kalouptsidi, Papageorgiou (2020), *Geography, Transportation, and Endogenous Trade Costs* (uses AIS at scale)
- RS Metrics and Orbital Insight methodology blog posts (read once)
- `geopandas`, `pyarrow`, `duckdb` docs for the geospatial + columnar bits

### Line target

400 total across all files.

### Sample README opening

```markdown
# AIS tanker dwell times -> WTI

Free public AIS pings to weekly Houston tanker dwell time, regressed on
WTI front-month returns.

## Result
Spearman IC of 0.11 between weekly median dwell hours and 5-day forward
WTI return, 2022-2023 (n=104 weeks, p=0.26). Event study around
top/bottom decile dwell-time weeks shows a 0.4% return spread over the
following 10 trading days, not significant at 5%.

In other words: the signal is in the right direction (more crude sitting
in port -> price weakness) but does not survive even loose significance
testing on two years of data. The pipeline is the deliverable.

## Run
    pip install pyarrow geopandas yfinance pandas duckdb
    python data.py       # ~20 min, ~6 GB of parquet output
    python features.py   # ~5 min
    python signal.py     # <1 min, prints table
```

---

## Project 2: NYC TLC distributed pipeline + model

```
experiments/tlc-distributed/
  README.md
  ingest.py            # download TLC parquet, hive-partition
  process_polars.py    # feature pipeline in Polars
  process_dask.py      # same pipeline in Dask
  process_duckdb.py    # same pipeline as one SQL query
  bench.py             # wall-clock + peak RSS comparison
  train.py             # Ray Train + XGBoost on the features
  data/                # gitignored
```

### Scope

Take NYC TLC yellow + green + FHV trip records for 2019-2024
(~500 million trips, ~30 GB compressed parquet). Build a tip-percentage
prediction model end-to-end. Implement the same feature pipeline three
ways:

1. **Polars** — single-node, lazy frames, streaming engine
2. **Dask** — out-of-core dataframes
3. **DuckDB** — one SQL query against the parquet partitions

Benchmark wall-clock and peak RSS on a fixed query: weekly trip count,
median fare, median tip percent, joined with taxi-zone metadata, grouped
by zone and weekday. Be honest about when each tool wins.

Then train an XGBoost tip-percentage model with `Ray Train` + `xgboost-ray`
across 4 local workers. Report RMSE on a time-ordered holdout (train on
2019-2023, test on 2024). Target RMSE < 4%.

### Files

- `ingest.py` (~60 lines)
  - Download yellow + green + FHV parquet for 2019-2024 from
    `s3://nyc-tlc/` (public, no creds).
  - Re-partition to `data/trips/year=YYYY/month=MM/type=T/*.parquet`.
- `process_polars.py` (~80 lines)
  - Lazy scan of partitioned parquet, derive `tip_pct`, `trip_minutes`,
    `pickup_hour`, `pickup_dow`, join taxi-zone csv, group + aggregate.
- `process_dask.py` (~80 lines)
  - Same pipeline using `dask.dataframe` with explicit partitioning.
- `process_duckdb.py` (~50 lines)
  - One CTE-heavy SQL query over the parquet glob with a join to the
    zone lookup table.
- `bench.py` (~80 lines)
  - Run each pipeline three times, record wall-clock + peak RSS via
    `psutil`, output a markdown table.
- `train.py` (~150 lines)
  - `ray.init()` local, 4 actors, `xgboost-ray` with early stopping,
    time-ordered split, print RMSE + feature importance.
- `README.md` (60-70 lines)

### References

- Polars, Dask, DuckDB docs (read the "out-of-core" sections of each)
- `xgboost-ray` README + the Ray Train docs
- NYC TLC trip-record data dictionary
- Marc Bessin and Wes McKinney's posts comparing Arrow-backed engines

### Line target

600 total.

### Expected honest result

Roughly:

- **DuckDB**: fastest for the analytical query, lowest memory (single
  SQL pass, no intermediate frames)
- **Polars (lazy)**: close second, slightly higher memory, better when
  the pipeline is imperative rather than a single SELECT
- **Dask**: slowest for this query (overhead of distributed scheduling
  on a single node), but only one that survives if the input outgrows
  a single machine

State the actual ratios. The point of this project is the honest
comparison, not crowning a winner.

### Sample README opening

```markdown
# TLC distributed pipeline + tip-percentage model

500M NYC taxi trips, three data engines on the same query, one Ray
Train XGBoost model on the result.

## Result: weekly fare + tip aggregate (96 weeks, 263 zones, 6 years)
| engine  | wall-clock | peak RSS |
|---------|-----------:|---------:|
| DuckDB  |   42 s     |  3.1 GB  |
| Polars  |   58 s     |  5.4 GB  |
| Dask    |  127 s     |  4.2 GB  |

## Result: tip-percentage model
RMSE 3.7% on 2024 holdout (n=78M), trained with Ray Train across 4
actors in 12 min on M2 Max. Top features: trip distance, hour of day,
pickup zone, payment type.
```

---

## Build order and time

| # | Project | Time | Why this order |
| --- | --- | --- | --- |
| 1 | AIS WTI signal | 3-4 days | Stand-alone, exercises geospatial + financial joins + honest backtest methodology |
| 2 | TLC distributed | 3-4 days | Different stack, no conceptual overlap; produces a reusable opinion on Polars vs Dask vs DuckDB |

These can run in either order. AIS first if you want the signal-research
muscle; TLC first if you want the data-engineering muscle.

## Commit conventions

One commit per working version, optional cleanup commit. Sample:

```
experiments/ais-wti-signal: Houston tanker dwell -> WTI, IC 0.11 (p=0.26)
experiments/tlc-distributed: 500M trips, DuckDB/Polars/Dask shootout + Ray Train RMSE 3.7%
```

## What to update outside `experiments/`

After both are done:

- Top-level `README.md` Layout table already lists `experiments/` — no change needed.
- `experiments/README.md` is currently a placeholder; replace with a
  table linking to each subdirectory and a one-line result, same shape
  as `implementations/README.md`.
