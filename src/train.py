import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.preprocessing import generate_sample_data, create_point_in_time_features

def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

def train_and_log():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    df = generate_sample_data(n_weeks=52)
    df_features = create_point_in_time_features(df)
    
    X = df_features[["sales_lag_1"]]
    y = df_features["sales"]
    
    train_size = int(len(df_features) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    mlflow.set_experiment("fmcg_demand_forecasting")
    
    with mlflow.start_run(run_name="lightgbm_tweedie_run"):
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "learning_rate": 0.05,
            "n_estimators": 50,
            "random_state": 42,
            "verbose": -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        wape_score = calculate_wape(y_test.values, predictions)
        
        mlflow.log_params(params)
        mlflow.log_metric("wape", wape_score)
        mlflow.lightgbm.log_model(model, artifact_path="model")
        
        print(f"Модель успішно навчена! Результат WAPE: {wape_score:.4f}")

if __name__ == "__main__":
    train_and_log()
