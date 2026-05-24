"""Feature pipeline in Polars (lazy). Output: weekly (PU zone, DO zone, dow) aggregates."""

import random
import time
from pathlib import Path

import polars as pl

random.seed(0)

DATA_DIR = Path(__file__).parent / "data"
TRIPS_GLOB = str(DATA_DIR / "trips" / "year=*" / "month=*" / "type=*" / "part.parquet")
LOOKUP_PATH = Path(__file__).parent / "taxi_zone_lookup.csv"
OUT_PATH = DATA_DIR / "out_polars.parquet"


def build_pipeline() -> pl.LazyFrame:
    lookup = pl.scan_csv(LOOKUP_PATH).select(
        pl.col("LocationID").cast(pl.Int64),
        pl.col("Zone").alias("zone"),
    )
    # hive_partitioning pulls year/month/type from the path so we don't have to parse them
    trips = pl.scan_parquet(
        TRIPS_GLOB, hive_partitioning=True,
        cast_options=pl.ScanCastOptions(integer_cast="upcast"),
    )

    feats = trips.with_columns(
        ((pl.col("dropoff") - pl.col("pickup")).dt.total_seconds() / 60.0).alias("trip_minutes"),
        pl.col("pickup").dt.hour().alias("pickup_hour"),
        pl.col("pickup").dt.weekday().alias("pickup_dow"),
        pl.col("pickup").dt.truncate("1w").alias("week"),
        # tip_pct only well-defined for yellow + green (fhv has no fares); guard div-by-zero
        pl.when((pl.col("type") != "fhv") & (pl.col("fare_amount") > 0))
          .then(pl.col("tip_amount") / pl.col("fare_amount"))
          .otherwise(None)
          .alias("tip_pct"),
    ).filter(
        (pl.col("trip_minutes") > 0) & (pl.col("trip_minutes") < 24 * 60)
        & pl.col("PULocationID").is_not_null() & pl.col("DOLocationID").is_not_null()
    )

    pu = lookup.rename({"LocationID": "PULocationID", "zone": "pu_zone"})
    do = lookup.rename({"LocationID": "DOLocationID", "zone": "do_zone"})
    joined = feats.join(pu, on="PULocationID", how="left").join(do, on="DOLocationID", how="left")

    return joined.group_by(["pu_zone", "do_zone", "pickup_dow", "week"]).agg(
        pl.len().alias("trip_count"),
        pl.col("fare_amount").median().alias("median_fare"),
        pl.col("tip_pct").median().alias("median_tip_pct"),
    ).sort(["week", "pu_zone", "do_zone", "pickup_dow"])


def main():
    t0 = time.time()
    df = build_pipeline().collect(engine="streaming")
    df.write_parquet(OUT_PATH, compression="snappy")
    dt = time.time() - t0
    print(f"polars: {df.height:,} rows, {OUT_PATH.stat().st_size/1e6:.1f} MB, {dt:.1f}s")
    return df


if __name__ == "__main__":
    out = main()
    assert OUT_PATH.exists()
    assert out.height > 0
    assert {"pu_zone", "do_zone", "pickup_dow", "week", "trip_count",
            "median_fare", "median_tip_pct"} == set(out.columns)
