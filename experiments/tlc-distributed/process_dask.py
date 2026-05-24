"""Same feature pipeline in Dask."""

import random
import time
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

random.seed(0)

DATA_DIR = Path(__file__).parent / "data"
TRIPS_GLOB = str(DATA_DIR / "trips" / "year=*" / "month=*" / "type=*" / "part.parquet")
LOOKUP_PATH = Path(__file__).parent / "taxi_zone_lookup.csv"
OUT_PATH = DATA_DIR / "out_dask.parquet"


def build_pipeline():
    trips = dd.read_parquet(TRIPS_GLOB, engine="pyarrow")
    # hive partitions come back as categoricals; tip_pct branch needs 'type' as a string
    trips["type"] = trips["type"].astype("string")
    trips = trips[trips["PULocationID"].notnull() & trips["DOLocationID"].notnull()]
    trips["trip_minutes"] = (trips["dropoff"] - trips["pickup"]).dt.total_seconds() / 60.0
    trips = trips[(trips["trip_minutes"] > 0) & (trips["trip_minutes"] < 1440)]
    trips["pickup_hour"] = trips["pickup"].dt.hour
    trips["pickup_dow"] = trips["pickup"].dt.weekday + 1
    trips["week"] = trips["pickup"].dt.to_period("W").dt.start_time
    trips["tip_pct"] = (trips["tip_amount"] / trips["fare_amount"]).where(
        (trips["type"] != "fhv") & (trips["fare_amount"] > 0)
    )

    lookup = pd.read_csv(LOOKUP_PATH)[["LocationID", "Zone"]]
    pu = lookup.rename(columns={"LocationID": "PULocationID", "Zone": "pu_zone"})
    do_ = lookup.rename(columns={"LocationID": "DOLocationID", "Zone": "do_zone"})
    trips = trips.merge(pu, on="PULocationID", how="left").merge(do_, on="DOLocationID", how="left")

    # median is non-trivial in Dask: groupby().agg('median') is supported but expensive — that's
    # the point of the bench. We expose the same semantics, not a faster approximation.
    agg = trips.groupby(["pu_zone", "do_zone", "pickup_dow", "week"], dropna=False).agg(
        trip_count=("pickup", "size"),
        median_fare=("fare_amount", "median"),
        median_tip_pct=("tip_pct", "median"),
    ).reset_index()
    return agg


def main():
    t0 = time.time()
    agg = build_pipeline().compute()
    agg = agg.sort_values(["week", "pu_zone", "do_zone", "pickup_dow"]).reset_index(drop=True)
    agg.to_parquet(OUT_PATH, compression="snappy", index=False)
    dt = time.time() - t0
    print(f"dask: {len(agg):,} rows, {OUT_PATH.stat().st_size/1e6:.1f} MB, {dt:.1f}s")
    return agg


if __name__ == "__main__":
    out = main()
    assert OUT_PATH.exists()
    assert len(out) > 0
    assert {"pu_zone", "do_zone", "pickup_dow", "week", "trip_count",
            "median_fare", "median_tip_pct"} == set(out.columns)
