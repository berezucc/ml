"""Download NYC TLC trip parquet and re-partition by year/month/type."""

import random
import shutil
import time
from pathlib import Path

import polars as pl
import requests

random.seed(0)

YEAR_START = 2023
YEAR_END = 2023
TYPES = ("yellow", "green", "fhv")
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
TRIPS_DIR = DATA_DIR / "trips"
LOOKUP_PATH = Path(__file__).parent / "taxi_zone_lookup.csv"


def download(url: str, dest: Path) -> int:
    """Stream-download URL to dest, returns bytes written. Skips if dest exists."""
    if dest.exists():
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    tmp.rename(dest)
    return dest.stat().st_size


def normalize(raw: Path, kind: str) -> pl.DataFrame:
    """Read raw parquet, rename to canonical columns, drop the rest."""
    df = pl.read_parquet(raw)
    if kind == "yellow":
        return df.rename({"tpep_pickup_datetime": "pickup", "tpep_dropoff_datetime": "dropoff"}).select(
            "pickup", "dropoff", "PULocationID", "DOLocationID", "passenger_count",
            "trip_distance", "fare_amount", "tip_amount", "payment_type",
        )
    if kind == "green":
        return df.rename({"lpep_pickup_datetime": "pickup", "lpep_dropoff_datetime": "dropoff"}).select(
            "pickup", "dropoff", "PULocationID", "DOLocationID", "passenger_count",
            "trip_distance", "fare_amount", "tip_amount",
            pl.col("payment_type").cast(pl.Int64),
        )
    # fhv: no fares, different column casing
    return df.rename({
        "pickup_datetime": "pickup", "dropOff_datetime": "dropoff",
        "PUlocationID": "PULocationID", "DOlocationID": "DOLocationID",
    }).select(
        "pickup", "dropoff",
        pl.col("PULocationID").cast(pl.Int64, strict=False),
        pl.col("DOLocationID").cast(pl.Int64, strict=False),
        pl.lit(None, dtype=pl.Float64).alias("passenger_count"),
        pl.lit(None, dtype=pl.Float64).alias("trip_distance"),
        pl.lit(None, dtype=pl.Float64).alias("fare_amount"),
        pl.lit(None, dtype=pl.Float64).alias("tip_amount"),
        pl.lit(None, dtype=pl.Int64).alias("payment_type"),
    )


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRIPS_DIR.mkdir(parents=True, exist_ok=True)
    download(LOOKUP_URL, LOOKUP_PATH)
    total_bytes = 0
    total_rows = 0
    t0 = time.time()
    for year in range(YEAR_START, YEAR_END + 1):
        for month in range(1, 13):
            for kind in TYPES:
                url = f"{BASE_URL}/{kind}_tripdata_{year}-{month:02d}.parquet"
                raw = RAW_DIR / f"{kind}_{year}-{month:02d}.parquet"
                out_dir = TRIPS_DIR / f"year={year}" / f"month={month:02d}" / f"type={kind}"
                out = out_dir / "part.parquet"
                if out.exists():
                    total_rows += pl.scan_parquet(out).select(pl.len()).collect().item()
                    total_bytes += out.stat().st_size
                    continue
                size = download(url, raw)
                total_bytes += size
                df = normalize(raw, kind)
                out_dir.mkdir(parents=True, exist_ok=True)
                df.write_parquet(out, compression="snappy")
                total_rows += df.height
                print(f"{kind} {year}-{month:02d}: {df.height:>10,} rows, {size/1e6:6.1f} MB raw")
    print(f"\ntotal: {total_rows:,} rows, {total_bytes/1e9:.2f} GB downloaded, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
    assert LOOKUP_PATH.exists()
    assert TRIPS_DIR.exists()
    parts = list(TRIPS_DIR.rglob("part.parquet"))
    expected = (YEAR_END - YEAR_START + 1) * 12 * len(TYPES)
    assert len(parts) == expected, f"got {len(parts)} parts, expected {expected}"
    print(f"ok: {len(parts)} partitions under {TRIPS_DIR}")
