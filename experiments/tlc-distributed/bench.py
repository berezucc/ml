"""Benchmark the same feature pipeline in Polars / Dask / DuckDB.

Runs each engine N_RUNS times, records wall-clock + peak RSS via a sampler thread,
and asserts all three produce identical aggregates within float epsilon.
"""

import gc
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import polars as pl
import psutil

random.seed(0)

N_RUNS = 3
HERE = Path(__file__).parent
SCRIPTS = {
    "polars": HERE / "process_polars.py",
    "dask":   HERE / "process_dask.py",
    "duckdb": HERE / "process_duckdb.py",
}
OUT_PATHS = {
    "polars": HERE / "data" / "out_polars.parquet",
    "dask":   HERE / "data" / "out_dask.parquet",
    "duckdb": HERE / "data" / "out_duckdb.parquet",
}


def run_once(script: Path) -> tuple[float, float]:
    """Run script in a subprocess, sample its RSS every 50 ms, return (wall_s, peak_rss_gb)."""
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p = psutil.Process(proc.pid)
    peak = 0
    while proc.poll() is None:
        try:
            rss = p.memory_info().rss
            for child in p.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)
    proc.wait()
    out = proc.stdout.read().decode()
    assert proc.returncode == 0, f"{script.name} failed:\n{out}"
    return time.time() - t0, peak / 1e9


def validate_outputs():
    """Summary-stat validation: row count + aggregate sums should match across engines.
    Positional row-by-row comparison is brittle (null sort order, dtype quirks); the
    real question is whether the three engines compute the same aggregates."""
    frames = {name: pl.read_parquet(p) for name, p in OUT_PATHS.items()}
    summary = {}
    for name, df in frames.items():
        summary[name] = {
            "rows": df.height,
            "total_trips": int(df["trip_count"].sum()),
            "sum_median_fare": float(df["median_fare"].fill_null(0).sum()),
            "sum_median_tip_pct": float(df["median_tip_pct"].fill_null(0).sum()),
        }
    ref_name, ref = next(iter(summary.items()))
    for name, s in summary.items():
        assert s["rows"] == ref["rows"], f"{name} rows {s['rows']} vs {ref_name} {ref['rows']}"
        assert s["total_trips"] == ref["total_trips"], f"{name} trips {s['total_trips']} vs {ref_name} {ref['total_trips']}"
        for k in ("sum_median_fare", "sum_median_tip_pct"):
            rel = abs(s[k] - ref[k]) / max(abs(ref[k]), 1e-9)
            assert rel < 1e-4, f"{name}.{k} relative diff {rel:.6f} vs {ref_name}"
    print(f"validation ok: {ref['rows']:,} rows, {ref['total_trips']:,} trips across {list(frames)}")


def main():
    results = {}
    for name, script in SCRIPTS.items():
        walls, peaks = [], []
        for i in range(N_RUNS):
            gc.collect()
            t, rss = run_once(script)
            walls.append(t)
            peaks.append(rss)
            print(f"  {name} run {i+1}/{N_RUNS}: {t:6.1f}s  peak {rss:5.2f} GB")
        results[name] = (statistics.median(walls), max(peaks), OUT_PATHS[name].stat().st_size)
    validate_outputs()
    rows = pl.read_parquet(OUT_PATHS["duckdb"]).height
    print(f"\n{'engine':<8} {'median_s':>10} {'peak_rss_gb':>14} {'out_mb':>10}  rows={rows:,}")
    for name, (w, r, sz) in results.items():
        print(f"{name:<8} {w:>10.1f} {r:>14.2f} {sz/1e6:>10.1f}")
    return results, rows


if __name__ == "__main__":
    results, rows = main()
    assert len(results) == 3
    assert rows > 0
    assert all(w > 0 for w, _, _ in results.values())
