import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import lightgbm as lgb
import mlflow.lightgbm
import numpy as np
import pandas as pd

from src.preprocessing import create_point_in_time_features


def load_artifacts(
    model_path: str, features_path: Optional[str] = None
) -> tuple[lgb.LGBMRegressor, List[str]]:
    """Loads trained model and expected feature names from MLflow or local files."""
    if model_path.startswith("runs:/") or model_path.startswith("models:/"):
        # Load directly from MLflow
        model = mlflow.lightgbm.load_model(model_path)
        if features_path and Path(features_path).exists():
            with open(features_path, "r") as f:
                feature_cols = json.load(f)
        else:
            feature_cols = model.booster_.feature_name()
    else:
        # Load from local directory (e.g., ./artifacts)
        local_dir = Path(model_path)
        feat_file = Path(features_path) if features_path else local_dir / "features.json"
        
        with open(feat_file, "r") as f:
            feature_cols = json.load(f)

        # In MLflow artifacts, the model directory contains the booster
        model_artifact_path = local_dir / "model"
        if model_artifact_path.exists():
            model = mlflow.lightgbm.load_model(str(model_artifact_path))
        else:
            # Fallback if raw booster or pkl was saved directly
            import joblib
            model = joblib.load(local_dir / "model.pkl")

    return model, feature_cols


def generate_recursive_forecast(
    df_history: pd.DataFrame,
    model: lgb.LGBMRegressor,
    feature_cols: List[str],
    horizon: int = 4,
    time_col: str = "week",
    target_col: str = "target",
    id_col: str = "series_id",
) -> pd.DataFrame:
    """Generates out-of-sample recursive forecast for H future weeks.
    
    Parameters:
    ----------
    df_history : pd.DataFrame
        Historical dataset (needs at least 52 previous weeks for annual lags).
    model : lgb.LGBMRegressor
        Trained forecasting model.
    feature_cols : List[str]
        Exact ordered list of feature column names.
    horizon : int
        Number of weeks to forecast into the future.
    """
    current_history = df_history.copy()
    last_known_week = current_history[time_col].max()
    all_series_ids = current_history[id_col].unique()

    forecast_records = []

    for step in range(1, horizon + 1):
        target_week = last_known_week + step

        # 1. Recalculate lags and moving averages including previous predictions
        df_with_features = create_point_in_time_features(
            current_history,
            lags=[1, 2, 4, 52],
            windows=[4, 8, 12]
        )

        # 2. Extract current step row for each series
        # In recursive forecasting, step features sit in the latest available historical week
        step_features = (
            df_with_features.sort_values([id_col, time_col])
            .groupby(id_col)
            .last()
            .reset_index()
        )

        X_step = step_features[feature_cols]

        # 3. Predict step sales
        raw_predictions = model.predict(X_step)
        # Demand cannot be negative
        step_predictions = np.clip(raw_predictions, a_min=0.0, a_max=None)

        # 4. Save predictions
        step_result = pd.DataFrame({
            id_col: step_features[id_col],
            time_col: target_week,
            "forecast": step_predictions
        })
        forecast_records.append(step_result)

        # 5. Append step prediction as simulated actuals into history for next step lags
        simulated_step = step_result.rename(columns={"forecast": target_col})
        # Carry forward static / future known attributes (e.g., price, is_promo)
        if "price" in current_history.columns:
            simulated_step["price"] = step_features["price"].values
        if "is_promo" in current_history.columns:
            simulated_step["is_promo"] = 0

        current_history = pd.concat([current_history, simulated_step], ignore_index=True)

    final_forecast = pd.concat(forecast_records, ignore_index=True)
    return final_forecast.sort_values([id_col, time_col]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Generate future sales forecast.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to full historical Parquet file",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./artifacts",
        help="Directory containing saved model or MLflow run URI",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to features.json (defaults to model-dir/features.json)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="Forecast horizon in weeks (H)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./forecasts.parquet",
        help="Destination path for generated predictions",
    )

    args = parser.parse_args()

    # Load artifacts and historical data
    model, feature_cols = load_artifacts(args.model_dir, args.features_path)
    df_history = pd.read_parquet(args.data_path)

    print(f"Loaded model with {len(feature_cols)} features.")
    print(f"Generating recursive forecast for {len(df_history['series_id'].unique())} series, horizon={args.horizon} weeks...")

    forecast_df = generate_recursive_forecast(
        df_history=df_history,
        model=model,
        feature_cols=feature_cols,
        horizon=args.horizon,
    )

    # Save outputs
    output_file = Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_parquet(output_file, index=False)

    print(f"Forecast successfully saved to {output_file}.")
    print("\nForecast sample:")
    print(forecast_df.head(8))


if __name__ == "__main__":
    main()