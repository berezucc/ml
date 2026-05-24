"""Same feature pipeline expressed as one DuckDB SQL query."""

import random
import time
from pathlib import Path

import duckdb

random.seed(0)

DATA_DIR = Path(__file__).parent / "data"
TRIPS_GLOB = str(DATA_DIR / "trips" / "year=*" / "month=*" / "type=*" / "part.parquet")
LOOKUP_PATH = Path(__file__).parent / "taxi_zone_lookup.csv"
OUT_PATH = DATA_DIR / "out_duckdb.parquet"

QUERY = f"""
COPY (
  WITH trips AS (
    SELECT
      pickup, dropoff, PULocationID, DOLocationID,
      fare_amount, tip_amount, type,
      date_diff('second', pickup, dropoff) / 60.0     AS trip_minutes,
      hour(pickup)                                    AS pickup_hour,
      isodow(pickup)                                  AS pickup_dow,
      date_trunc('week', pickup)                      AS week,
      CASE WHEN type <> 'fhv' AND fare_amount > 0
           THEN tip_amount / fare_amount END          AS tip_pct
    FROM read_parquet('{TRIPS_GLOB}', hive_partitioning=true)
    WHERE PULocationID IS NOT NULL AND DOLocationID IS NOT NULL
  ),
  lookup AS (SELECT LocationID, Zone FROM read_csv_auto('{LOOKUP_PATH}'))
  SELECT pu.Zone AS pu_zone, dz.Zone AS do_zone, t.pickup_dow, t.week,
         count(*)               AS trip_count,
         median(t.fare_amount)  AS median_fare,
         median(t.tip_pct)      AS median_tip_pct
  FROM trips t
  LEFT JOIN lookup pu ON t.PULocationID = pu.LocationID
  LEFT JOIN lookup dz ON t.DOLocationID = dz.LocationID
  WHERE t.trip_minutes > 0 AND t.trip_minutes < 1440
  GROUP BY 1, 2, 3, 4
  ORDER BY week, pu_zone, do_zone, pickup_dow
) TO '{OUT_PATH}' (FORMAT PARQUET, COMPRESSION SNAPPY);
"""


def main():
    t0 = time.time()
    con = duckdb.connect()
    # Default thread count = cores; we don't pin so this is fair against Polars/Dask
    con.execute(QUERY)
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{OUT_PATH}')").fetchone()[0]
    dt = time.time() - t0
    print(f"duckdb: {rows:,} rows, {OUT_PATH.stat().st_size/1e6:.1f} MB, {dt:.1f}s")
    return rows


if __name__ == "__main__":
    n = main()
    assert OUT_PATH.exists()
    assert n > 0
