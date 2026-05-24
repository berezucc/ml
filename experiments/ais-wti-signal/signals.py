"""Join weekly tanker features with WTI front-month, compute IC + event study.

Reads `data/weekly.parquet`; downloads WTI (CL=F) from yfinance.
Prints a markdown table; no plots.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

random.seed(0)
np.random.seed(0)

DATA_DIR = Path(__file__).parent / "data"
WEEKLY_PATH = DATA_DIR / "weekly.parquet"
JOIN_PATH = DATA_DIR / "joined.parquet"

# Forward return horizon. 5 trading days = ~1 week.
FWD_DAYS = 5

# WTI continuous front-month futures.
WTI_TICKER = "CL=F"

FEATURES = ["arrivals", "median_dwell_h", "total_dwell_h", "p90_dwell_h", "unique_vessels"]


def load_wti(start, end):
    # Buffer the window so 5-day forward return at the last week still has data.
    df = yf.download(WTI_TICKER, start=start, end=end + pd.Timedelta(days=20),
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize("UTC")
    df["log_ret_fwd"] = np.log(df["Close"].shift(-FWD_DAYS) / df["Close"])
    return df


def build_panel(weekly, wti):
    """Lag features by 1 week (no lookahead) and join to Friday-close fwd return."""
    w = weekly.copy()
    w["week"] = pd.to_datetime(w["week"], utc=True)
    w = w.sort_values("week").reset_index(drop=True)

    # Lag every feature by 1 week.
    for col in FEATURES:
        w[col] = w[col].shift(1)
    w = w.dropna().reset_index(drop=True)

    # Reference WTI close = last trading day in (week, week+7d). Use Friday-or-prior.
    week_close = []
    for ts in w["week"]:
        end = ts + pd.Timedelta(days=7)
        slc = wti.loc[ts:end - pd.Timedelta(seconds=1)]
        if len(slc):
            week_close.append(slc.iloc[-1])
        else:
            week_close.append(pd.Series({"Close": np.nan, "log_ret_fwd": np.nan}))
    closes = pd.DataFrame(week_close).reset_index(drop=True)
    w["wti_close"] = closes["Close"].values
    w["log_ret_fwd"] = closes["log_ret_fwd"].values
    w = w.dropna(subset=["log_ret_fwd"]).reset_index(drop=True)
    return w


def ic_table(panel):
    rows = []
    y = panel["log_ret_fwd"].to_numpy()
    for f in FEATURES:
        x = panel[f].to_numpy()
        p_r, p_p = stats.pearsonr(x, y)
        s_r, s_p = stats.spearmanr(x, y)
        rows.append((f, p_r, p_p, s_r, s_p))
    return pd.DataFrame(rows, columns=["feature", "pearson_r", "pearson_p", "spearman_r", "spearman_p"])


def event_study(panel, feature="total_dwell_h"):
    """Top-decile vs bottom-decile weeks: mean fwd return + t-test."""
    q_hi = panel[feature].quantile(0.9)
    q_lo = panel[feature].quantile(0.1)
    hi = panel.loc[panel[feature] >= q_hi, "log_ret_fwd"].to_numpy()
    lo = panel.loc[panel[feature] <= q_lo, "log_ret_fwd"].to_numpy()
    if len(hi) < 2 or len(lo) < 2:
        return None
    spread = hi.mean() - lo.mean()
    t_stat, t_p = stats.ttest_ind(hi, lo, equal_var=False)
    return {
        "feature": feature,
        "n_hi": int(len(hi)),
        "n_lo": int(len(lo)),
        "mean_fwd_hi": float(hi.mean()),
        "mean_fwd_lo": float(lo.mean()),
        "spread": float(spread),
        "t_stat": float(t_stat),
        "p_value": float(t_p),
    }


def main():
    if not WEEKLY_PATH.exists():
        raise SystemExit(f"missing {WEEKLY_PATH}; run features.py first")
    weekly = pd.read_parquet(WEEKLY_PATH)
    print(f"weekly rows: {len(weekly)} ({weekly['week'].min()} .. {weekly['week'].max()})")
    wti = load_wti(weekly["week"].min(), weekly["week"].max())
    print(f"WTI rows: {len(wti)} trading days")
    panel = build_panel(weekly, wti)
    panel.to_parquet(JOIN_PATH, index=False)
    print(f"panel rows: {len(panel)} (after 1-week lag + fwd-return join)")
    print()

    ic = ic_table(panel)
    print("## IC: weekly tanker feature (lag 1w) -> 5-day fwd WTI log return")
    print()
    print("| feature | pearson_r | pearson_p | spearman_r | spearman_p |")
    print("|---|---|---|---|---|")
    for _, r in ic.iterrows():
        print(f"| {r['feature']} | {r['pearson_r']:+.3f} | {r['pearson_p']:.3f} | "
              f"{r['spearman_r']:+.3f} | {r['spearman_p']:.3f} |")
    print(f"\nn_weeks = {len(panel)}")

    print("\n## Event study: top vs bottom decile weeks by total_dwell_h")
    es = event_study(panel, "total_dwell_h")
    if es is None:
        print("(not enough weeks for decile split)")
    else:
        print(f"- n_hi = {es['n_hi']}, n_lo = {es['n_lo']}")
        print(f"- mean fwd return | hi-dwell = {es['mean_fwd_hi']:+.4f}")
        print(f"- mean fwd return | lo-dwell = {es['mean_fwd_lo']:+.4f}")
        print(f"- spread (hi - lo) = {es['spread']:+.4f}")
        print(f"- Welch t = {es['t_stat']:+.3f}, p = {es['p_value']:.3f}")


if __name__ == "__main__":
    assert FWD_DAYS >= 1
    assert len(FEATURES) > 0
    if "--check" in sys.argv:
        print("config ok")
        sys.exit(0)
    main()
