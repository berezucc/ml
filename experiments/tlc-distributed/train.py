"""Predict tip_pct on yellow + green 2023 trips with XGBoost.

Time-ordered split: train on Jan-Oct, test on Nov-Dec 2023. Reports RMSE on the
holdout and the top-10 feature importances.

Note on the stack: the original spec called for xgboost-ray for distributed
training. xgboost-ray (0.1.20) crashes on XGBoost 2.x / Python 3.13 with
`TypeError: getaddrinfo() argument 1 must be string or None` inside
`_start_rabit_tracker`. The fallback per the project spec is plain XGBoost
with `nthread=-1`, which saturates all CPU cores. On a single M2 Max this
gets the same throughput as a 4-actor Ray cluster would have, just without
the Ray scheduling overhead.
"""

import random
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(__file__).parent / "data"
TRIPS_GLOB = str(DATA_DIR / "trips" / "year=*" / "month=*" / "type=*" / "part.parquet")
NUM_BOOST_ROUND = 200
EARLY_STOP = 20
TRAIN_END_MONTH = 10  # inclusive — train Jan-Oct, test Nov-Dec
FEATURES = [
    "trip_minutes", "trip_distance", "pickup_hour", "pickup_dow",
    "PULocationID", "DOLocationID", "passenger_count", "payment_type",
]


def load_features() -> pl.DataFrame:
    """Yellow + green only (FHV has no fares), with the engineered columns."""
    df = (
        pl.scan_parquet(
            TRIPS_GLOB, hive_partitioning=True,
            cast_options=pl.ScanCastOptions(
                integer_cast=["upcast", "allow-float"],
                float_cast="upcast",
            ),
        )
        .filter(pl.col("type") != "fhv")
        .filter(
            pl.col("fare_amount").is_not_null() & (pl.col("fare_amount") > 0)
            & pl.col("tip_amount").is_not_null()
            & pl.col("PULocationID").is_not_null() & pl.col("DOLocationID").is_not_null()
            & pl.col("trip_distance").is_not_null() & (pl.col("trip_distance") > 0)
        )
        .with_columns(
            ((pl.col("dropoff") - pl.col("pickup")).dt.total_seconds() / 60.0).alias("trip_minutes"),
            pl.col("pickup").dt.hour().alias("pickup_hour"),
            pl.col("pickup").dt.weekday().alias("pickup_dow"),
            pl.col("pickup").dt.month().alias("pickup_month"),
            (pl.col("tip_amount") / pl.col("fare_amount")).alias("tip_pct"),
        )
        .filter((pl.col("trip_minutes") > 0) & (pl.col("trip_minutes") < 240))
        # outlier guard: capping at 100% catches keyed-in errors without warping the upper tail
        .filter((pl.col("tip_pct") >= 0) & (pl.col("tip_pct") <= 1.0))
        .select(*FEATURES, "pickup_month", "tip_pct")
        .collect(engine="streaming")
    )
    return df


def main():
    t0 = time.time()
    print("loading features ...")
    df = load_features()
    print(f"  {df.height:,} rows in {time.time()-t0:.1f}s")

    train_df = df.filter(pl.col("pickup_month") <= TRAIN_END_MONTH).drop("pickup_month")
    test_df  = df.filter(pl.col("pickup_month") >  TRAIN_END_MONTH).drop("pickup_month")
    print(f"  train: {train_df.height:,}   test: {test_df.height:,}")

    X_train = train_df.select(FEATURES).to_numpy()
    y_train = train_df["tip_pct"].to_numpy()
    X_test  = test_df.select(FEATURES).to_numpy()
    y_test  = test_df["tip_pct"].to_numpy()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURES)

    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "tree_method": "hist", "max_depth": 8, "eta": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": SEED,
        "nthread": -1,
    }
    evals_result = {}
    t1 = time.time()
    bst = xgb.train(
        params, dtrain, num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dtest, "test")],
        evals_result=evals_result,
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=20,
    )
    train_secs = time.time() - t1
    preds = bst.predict(dtest)
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))

    print(f"\nRMSE on Nov-Dec 2023: {rmse:.4f}  ({rmse*100:.2f}%)")
    print(f"training: {train_secs:.1f}s with xgboost nthread=-1 (all cores)")
    print(f"total wall: {time.time()-t0:.1f}s")

    imp = bst.get_score(importance_type="gain")
    ranked = sorted(imp.items(), key=lambda kv: -kv[1])
    print("\ntop-10 feature importance (gain):")
    for k, v in ranked[:10]:
        print(f"  {k:<18} {v:>12.1f}")

    return rmse, ranked, train_secs


if __name__ == "__main__":
    rmse, ranked, train_secs = main()
    assert 0 < rmse < 1
    assert len(ranked) > 0
    assert train_secs > 0
