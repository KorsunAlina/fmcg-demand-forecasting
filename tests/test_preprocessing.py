import numpy as np
import pandas as pd
from src.preprocessing import create_point_in_time_features


def test_no_data_leakage_and_correct_lags():
    weeks = list(range(1, 60))

    raw_df = pd.DataFrame({
        "series_id": "series_001",
        "week": weeks,
        "target": [float(w * 10) for w in weeks],
        "price": 50.0,
        "is_promo": 0,
    })

    df_processed_1 = create_point_in_time_features(
        raw_df, lags=[1, 2, 52], windows=[4]
    )

    # Corrupt target at week 55 to test for future data leakage
    raw_df_corrupted = raw_df.copy()
    raw_df_corrupted.loc[raw_df_corrupted["week"] == 55, "target"] = 9999.0

    df_processed_2 = create_point_in_time_features(
        raw_df_corrupted, lags=[1, 2, 52], windows=[4]
    )

    # 1. Verify lag correctness (lag_1 at week 55 must equal target at week 54)
    lag_1_val = df_processed_1.loc[
        df_processed_1["week"] == 55, "target_lag_1"
    ].values[0]
    actual_week_54 = raw_df.loc[raw_df["week"] == 54, "target"].values[0]

    assert lag_1_val == actual_week_54, (
        f"Expected lag_1={actual_week_54}, got {lag_1_val}"
    )

    # 2. Verify no data leakage (modifying week 55 target must not affect week 55 features)
    features_cols = ["target_lag_1", "rolling_mean_4", "rolling_median_4"]

    features_week_55_before = df_processed_1.loc[
        df_processed_1["week"] == 55, features_cols
    ].values[0]

    features_week_55_after = df_processed_2.loc[
        df_processed_2["week"] == 55, features_cols
    ].values[0]

    np.testing.assert_array_equal(
        features_week_55_before,
        features_week_55_after,
        err_msg="Data leakage detected: week 55 features changed after modifying week 55 target.",
    )


def test_groupby_isolation():
    """Verify that feature generation is strictly isolated per series_id."""
    raw_df = pd.DataFrame({
        "series_id": ["series_001"] * 55 + ["series_002"] * 55,
        "week": list(range(1, 56)) * 2,
        "target": [100.0] * 55 + [900.0] * 55,
    })

    df_processed = create_point_in_time_features(
        raw_df, lags=[1, 52], windows=[4]
    )

    # Ensure series_002 does not leak lag_1 from series_001
    s2_lag1 = df_processed.loc[
        (df_processed["series_id"] == "series_002")
        & (df_processed["week"] == 53),
        "target_lag_1",
    ].values[0]

    assert s2_lag1 == 900.0, (
        f"Groupby isolation failed: expected 900.0, got {s2_lag1}"
    )