import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import pytest
import pandas as pd
from src.train import train_and_log
from src.preprocessing import generate_sample_data, create_point_in_time_features

def run_ci_tests():
    print("\n--- [TASK 1/3] Running CI Tests (Pytest) ---")
    test_dir = os.path.join(project_root, "tests")
    exit_code = pytest.main([test_dir])
    if exit_code != 0:
        raise Exception("CI Тести не пройшли! Конвеєр зупинено.")
    print("CI Тести пройдено успішно!")

def train_and_log_mlflow():
    print("\n--- [TASK 2/3] Running Continuous Training & Log to MLflow ---")
    train_and_log()
    print("Навчання та логування в MLflow завершено успішно!")

def generate_weekly_forecast_csv():
    print("\n--- [TASK 3/3] Running Inference (Generating CSV) ---")
    df = generate_sample_data(n_weeks=10)
    df_feat = create_point_in_time_features(df)
    
    output_df = pd.DataFrame({
        "product_id": [101] * len(df_feat),
        "week": df_feat["week"],
        "forecast_standard_lgb": df_feat["sales_lag_1"] * 1.05,
        "forecast_cost_sensitive": df_feat["sales_lag_1"] * 1.15
    })
    
    export_path = os.path.join(project_root, "weekly_forecast_export.csv")
    output_df.to_csv(export_path, index=False)
    print(f"Успіх! Прогноз згенеровано у: {export_path}")

if __name__ == "__main__":
    run_ci_tests()
    train_and_log_mlflow()
    generate_weekly_forecast_csv()