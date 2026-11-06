import argparse
import json
import os
import sys
from pathlib import Path
import joblib

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from src.split import expanding_window_split
from src.preprocessing import create_point_in_time_features


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE)."""
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true))


def calculate_mase( y_true: np.ndarray, y_pred: np.ndarray, y_baseline_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Scaled Error (MASE) against a seasonal-naive baseline."""
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_baseline = np.mean(np.abs(y_true - y_baseline_pred))

    if mae_baseline == 0:
        return 0.0

    return float(mae_model / mae_baseline)


def train_and_log(args: argparse.Namespace) -> None:
    """Train LightGBM model, evaluate metrics, and log artifacts to MLflow."""
    # Resolve MLflow tracking URI from environment variable
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    # Load dataset
    df_raw = pd.read_parquet(args.data_path)

    # Generate lags and rolling features if they are not already present
    if "target_lag_1" not in df_raw.columns:
        df = create_point_in_time_features(
            df_raw, 
            lags=[1, 2, 4, 52], 
            windows=[4, 8, 12]
        )
    else:
        df = df_raw

    # Expanding-window split: train on history up to cutoff, test on horizon H
    _, train_df, test_df = next( expanding_window_split(df, cutoffs=[args.cutoff_week], horizon=args.horizon))

    # Select features (exclude identifiers and target)
    ignore_cols = ["series_id", "week", "target"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    # Seasonal-naive baseline: use 52-week lag (fallback to lag 1 if unavailable)
    baseline_col = (
        "target_lag_52" if "target_lag_52" in feature_cols else "target_lag_1"
    )
    y_baseline_pred = test_df[baseline_col].values

    # Model configuration and training
    custom_params = json.loads(args.hyperparameters) if args.hyperparameters else {}
    default_params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "random_state": 42,
        "verbose": -1,
    }
    default_params.update(custom_params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)

    wape_score = calculate_wape(y_test.values, predictions)
    mase_score = calculate_mase(y_test.values, predictions, y_baseline_pred)
    baseline_wape = calculate_wape(y_test.values, y_baseline_pred)

    print(
        f"Cutoff: {args.cutoff_week} | Horizon: {args.horizon} | "
        f"Train size: {len(X_train)} | Test size: {len(X_test)}"
    )
    print(
        f"Model WAPE: {wape_score:.4f} | MASE: {mase_score:.4f} | "
        f"Baseline WAPE: {baseline_wape:.4f}"
    )

    #  Save feature list locally
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "features.json"
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=4)

    # Save model locally for predict.py
    joblib.dump(model, output_dir / "model.pkl")

    # Log run to MLflow
    try:
        mlflow.set_experiment("fmcg_demand_forecasting")

        with mlflow.start_run(run_name=f"lightgbm_cutoff_{args.cutoff_week}"):
            mlflow.log_params(default_params)
            mlflow.log_params({
                "cutoff_week": args.cutoff_week,
                "horizon": args.horizon,
                "data_path": args.data_path,
            })
            mlflow.log_metrics({
                "wape": wape_score,
                "mase": mase_score,
                "baseline_wape": baseline_wape,
            })
            mlflow.log_artifact(str(features_path), artifact_path="model")
            mlflow.lightgbm.log_model(model, artifact_path="model")
    except Exception as e:
        print(f"MLflow logging skipped or failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM forecasting model.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to Parquet dataset",
    )
    parser.add_argument(
        "--cutoff-week",
        type=int,
        required=True,
        help="Cutoff week for train/test split",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="Forecast horizon in weeks",
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default="{}",
        help="JSON string of hyperparameters",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./output",
        help="Directory to save output artifacts",
    )

    args = parser.parse_args()
    train_and_log(args)