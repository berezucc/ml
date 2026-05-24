"""Build vessel visits from per-day AIS parquet, then weekly aggregate features.

Reads `data/parquet/*.parquet` (from data.py).
Writes `data/visits.parquet` and `data/weekly.parquet`.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

random.seed(0)
np.random.seed(0)

DATA_DIR = Path(__file__).parent / "data"
PARQUET_DIR = DATA_DIR / "parquet"
VISITS_PATH = DATA_DIR / "visits.parquet"
WEEKLY_PATH = DATA_DIR / "weekly.parquet"

# Houston Ship Channel + Galveston Bay polygon. Convex hexagon over the
# tankering area: Bolivar Roads at the S, Barbours Cut / Houston turning
# basin at the NW, Trinity Bay E shoreline at the NE. Coarser than the
# actual channel but excludes the open Gulf and inland Houston, which is
# what matters for the dwell-time signal. Coords are (lon, lat).
HSC_POLYGON = Polygon([
    (-94.80, 29.34),  # SW corner near Bolivar / Texas City
    (-94.60, 29.34),  # SE corner, Gulf entrance E
    (-94.60, 29.62),  # E shoreline of bay
    (-94.85, 29.78),  # NE upper bay
    (-95.27, 29.78),  # NW Port of Houston turning basin
    (-95.05, 29.55),  # W shoreline ~ Kemah / La Porte
])

# Gap that breaks a single visit into two. If a tanker leaves the polygon for
# more than this many hours, the next entry is a new visit.
VISIT_GAP_HOURS = 1.0

# Drop visits shorter than this — likely GPS noise / channel transits.
MIN_VISIT_HOURS = 1.0


def _points_in_polygon(lon, lat, poly):
    """Vectorised point-in-polygon using shapely 2.x via numpy."""
    # shapely 2.x: vectorised contains via STRtree is overkill; use prep + map.
    from shapely import contains_xy
    return contains_xy(poly, lon, lat)


def load_pings():
    files = sorted(PARQUET_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {PARQUET_DIR}; run data.py first")
    parts = []
    for f in files:
        df = pd.read_parquet(f)
        if len(df):
            parts.append(df)
    if not parts:
        raise SystemExit("no AIS pings after filter")
    df = pd.concat(parts, ignore_index=True)
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], utc=True)
    df = df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)
    print(f"loaded {len(df):,} tanker pings, {df['MMSI'].nunique():,} unique MMSI, "
          f"{df['BaseDateTime'].min()} .. {df['BaseDateTime'].max()}")
    return df


def detect_visits(pings):
    """For each MMSI, find contiguous in-polygon spans separated by >VISIT_GAP_HOURS."""
    inside = _points_in_polygon(pings["LON"].to_numpy(), pings["LAT"].to_numpy(), HSC_POLYGON)
    pings = pings.assign(inside=inside)
    in_pings = pings[pings["inside"]].copy()
    print(f"  in-polygon pings: {len(in_pings):,} ({100 * len(in_pings) / len(pings):.1f}%)")
    if len(in_pings) == 0:
        return pd.DataFrame(columns=["MMSI", "start", "end", "dwell_hours"])

    # Sequential visit id per MMSI: bump when gap > VISIT_GAP_HOURS.
    in_pings = in_pings.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)
    dt_h = in_pings.groupby("MMSI")["BaseDateTime"].diff().dt.total_seconds() / 3600.0
    new_visit = (dt_h.isna()) | (dt_h > VISIT_GAP_HOURS)
    in_pings["visit_id"] = new_visit.cumsum()

    visits = in_pings.groupby(["MMSI", "visit_id"]).agg(
        start=("BaseDateTime", "min"),
        end=("BaseDateTime", "max"),
        n_pings=("BaseDateTime", "size"),
    ).reset_index(drop=False)
    visits["dwell_hours"] = (visits["end"] - visits["start"]).dt.total_seconds() / 3600.0
    visits = visits[visits["dwell_hours"] >= MIN_VISIT_HOURS].reset_index(drop=True)
    print(f"  visits >= {MIN_VISIT_HOURS}h: {len(visits):,}")
    return visits


def weekly_aggregate(visits):
    """ISO-week bucket on visit start. Returns one row per week."""
    if len(visits) == 0:
        return pd.DataFrame()
    v = visits.copy()
    v["start"] = pd.to_datetime(v["start"], utc=True)
    # ISO week label: Monday of that week.
    v["week"] = v["start"].dt.tz_convert("UTC").dt.to_period("W-SUN").dt.start_time
    grp = v.groupby("week")
    weekly = grp.agg(
        arrivals=("dwell_hours", "size"),
        median_dwell_h=("dwell_hours", "median"),
        total_dwell_h=("dwell_hours", "sum"),
        p90_dwell_h=("dwell_hours", lambda s: float(np.quantile(s, 0.9))),
        unique_vessels=("MMSI", "nunique"),
    ).reset_index()
    return weekly


def main():
    pings = load_pings()
    visits = detect_visits(pings)
    visits.to_parquet(VISITS_PATH, index=False)
    print(f"wrote {VISITS_PATH} ({len(visits):,} visits)")
    weekly = weekly_aggregate(visits)
    weekly.to_parquet(WEEKLY_PATH, index=False)
    print(f"wrote {WEEKLY_PATH} ({len(weekly):,} weeks)")
    if len(weekly):
        print("\nweekly head:")
        print(weekly.head(10).to_string(index=False))
        print("\nweekly tail:")
        print(weekly.tail(5).to_string(index=False))


if __name__ == "__main__":
    # Self-check: polygon sanity.
    assert HSC_POLYGON.is_valid, "HSC polygon invalid"
    assert HSC_POLYGON.contains(Point(-94.95, 29.65)), "Houston Ship Channel test pt should be inside"
    assert not HSC_POLYGON.contains(Point(-94.0, 29.0)), "open-Gulf test pt should be outside"
    assert VISIT_GAP_HOURS > 0 and MIN_VISIT_HOURS > 0
    if "--check" in sys.argv:
        print("polygon ok; centroid =", HSC_POLYGON.centroid)
        sys.exit(0)
    main()
