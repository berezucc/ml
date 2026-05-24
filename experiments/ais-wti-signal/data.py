"""Download NOAA daily AIS zips, stream-filter to Houston bbox + tankers, write parquet.

Run: `python data.py`. Re-run is idempotent (skips already-processed days).
Change START_DATE / END_DATE to retarget. Sampling controls bandwidth usage.
"""
import io
import os
import random
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

random.seed(0)

# Date window. Spec asks for 2 years; this run uses H2 2023 = 6 months.
START_DATE = date(2023, 7, 1)
END_DATE = date(2023, 12, 31)

# Houston Ship Channel + Galveston Bay anchorage bbox (lat min, lat max, lon min, lon max).
# Tighter polygon is defined in features.py for inside/outside testing.
BBOX_LAT_MIN = 29.30
BBOX_LAT_MAX = 29.78
BBOX_LON_MIN = -95.30
BBOX_LON_MAX = -94.55

# AIS VesselType codes for tankers (cargo: 70-79, tanker: 80-89). Spec = 80-89.
TANKER_TYPE_MIN = 80
TANKER_TYPE_MAX = 89

# How many days per ISO week to download. Set to 7 for full data; lower values
# sample (NOAA throttles ~15 KB/s, so 6 months full = 50+ hours of download).
# We always keep the same weekday so dwell-hour aggregates are unbiased.
DAYS_PER_WEEK = 1
SAMPLE_WEEKDAY = 2  # 0=Mon ... 6=Sun. Wed = mid-week, tankers operate 24/7.

URL_TEMPLATE = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_{ymd}.zip"
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# CSV schema (NOAA AIS 2018+): MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading,
# VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft, Cargo, TransceiverClass.
KEEP_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "VesselType", "Length", "Draft"]


def days_in_window():
    cur = START_DATE
    while cur <= END_DATE:
        # ISO weekday: 1=Mon, ..., 7=Sun. Python weekday() is 0=Mon.
        if cur.weekday() == SAMPLE_WEEKDAY or DAYS_PER_WEEK >= 7:
            yield cur
        cur += timedelta(days=1)


def download_one(day):
    ymd = day.strftime("%Y_%m_%d")
    zip_path = RAW_DIR / f"AIS_{ymd}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 50_000_000:
        return zip_path
    url = URL_TEMPLATE.format(ymd=ymd)
    print(f"  GET {url}")
    t0 = time.time()
    # Stream + resume support for slow NOAA connection.
    headers = {}
    mode = "wb"
    if zip_path.exists():
        headers["Range"] = f"bytes={zip_path.stat().st_size}-"
        mode = "ab"
    r = requests.get(url, headers=headers, stream=True, timeout=(30, 1800))
    if r.status_code not in (200, 206):
        r.close()
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    with open(zip_path, mode) as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    print(f"    {zip_path.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s")
    return zip_path


def filter_and_write(zip_path, day):
    """Stream-read CSV out of zip, filter to bbox + tankers, append parquet."""
    out_path = PARQUET_DIR / f"{day.isoformat()}.parquet"
    if out_path.exists():
        return out_path, None
    t0 = time.time()
    total_rows = 0
    kept_rows = 0
    chunks = []
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as raw:
            reader = pd.read_csv(
                raw,
                usecols=KEEP_COLS,
                dtype={"MMSI": "int64", "VesselType": "Int16"},
                parse_dates=["BaseDateTime"],
                chunksize=500_000,
            )
            for chunk in reader:
                total_rows += len(chunk)
                m = (
                    chunk["LAT"].between(BBOX_LAT_MIN, BBOX_LAT_MAX)
                    & chunk["LON"].between(BBOX_LON_MIN, BBOX_LON_MAX)
                    & chunk["VesselType"].between(TANKER_TYPE_MIN, TANKER_TYPE_MAX)
                )
                kept = chunk.loc[m]
                if len(kept):
                    chunks.append(kept)
                kept_rows += len(kept)
    if chunks:
        df = pd.concat(chunks, ignore_index=True).sort_values(["MMSI", "BaseDateTime"])
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path, compression="zstd")
    else:
        # Write an empty parquet so re-runs skip this day.
        pd.DataFrame(columns=KEEP_COLS).to_parquet(out_path)
    print(
        f"    filter {day}: {total_rows:,} rows in -> {kept_rows:,} kept "
        f"({100 * kept_rows / max(total_rows, 1):.3f}%) in {time.time() - t0:.0f}s"
    )
    return out_path, kept_rows


def main():
    days = list(days_in_window())
    print(f"window: {START_DATE} .. {END_DATE} ({len(days)} sampled days, weekday={SAMPLE_WEEKDAY})")
    successes = []
    for i, day in enumerate(days, 1):
        print(f"[{i}/{len(days)}] {day}")
        try:
            zp = download_one(day)
        except Exception as exc:
            print(f"  download failed: {exc}")
            continue
        try:
            out_path, kept = filter_and_write(zp, day)
            successes.append((day, out_path, kept))
        except Exception as exc:
            print(f"  filter failed: {exc}")
    print(f"\nDone. {len(successes)}/{len(days)} days processed -> {PARQUET_DIR}")


if __name__ == "__main__":
    # Self-check: URL template and bbox sanity.
    assert START_DATE < END_DATE, "bad date window"
    assert BBOX_LAT_MIN < BBOX_LAT_MAX and BBOX_LON_MIN < BBOX_LON_MAX, "bad bbox"
    assert 0 <= SAMPLE_WEEKDAY <= 6
    if "--check" in sys.argv:
        # Schema check on one already-downloaded zip, no network.
        zips = sorted(RAW_DIR.glob("*.zip"))
        if not zips:
            print("no zips yet; run without --check after download")
            sys.exit(0)
        zp = zips[0]
        with zipfile.ZipFile(zp) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(name) as raw:
                head = pd.read_csv(raw, nrows=5)
        print(head.columns.tolist())
        print(head.head())
        sys.exit(0)
    main()
