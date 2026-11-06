import pandas as pd


def create_point_in_time_features(df: pd.DataFrame, lags: list[int] = [1, 2, 4, 52], windows:list[int] = [4, 8, 12]):

    df = df.copy()

    # Sorting by prduct and week
    df = df.sort_values(["series_id", "week"]).reset_index(drop=True)

    for lag in lags:
        df[f"target_lag_{lag}"] = df.groupby("series_id")["target"].shift(lag)

    # Rolling stats
    grouped_target = df.groupby("series_id")["target"]

    for window in windows:
        # Rolling mean
        df[f"rolling_mean_{window}"] = grouped_target.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        # Rolling median
        df[f"rolling_median_{window}"] = grouped_target.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).median()
        )

    max_lag = max(lags)
    df = df.dropna(subset=[f"target_lag_{max_lag}"]).reset_index(drop=True)

    return df