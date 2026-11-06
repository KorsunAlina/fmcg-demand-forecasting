import json
import os
from datetime import datetime, timedelta
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator

# Базові шляхи (використовуються для монтування томів Docker)
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
S3_BUCKET = "fmcg-demand-forecast-alina-2026"
ECR_IMAGE = "fmcg-forecasting:latest"

default_args = {
    "owner": "alina",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

def aggregate_metrics_fn():
    """Зчитує метрики з 4 фолдів і перевіряє порогову якість"""
    cutoffs = [88, 92, 96, 100]
    wapes = []
    
    for cutoff in cutoffs:
        metrics_path = f"{PROJECT_DIR}/model_output/fold_{cutoff}/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
                wapes.append(data.get("wape", 1.0))
        else:
            print(f"Warning: {metrics_path} не знайдено.")

    if not wapes:
        raise ValueError("Жоден файл метрик не знайдено!")

    mean_wape = sum(wapes) / len(wapes)
    print(f"=== БЕКТЕСТ ЗАВЕРШЕНО ===")
    print(f"Фолди WAPE: {wapes}")
    print(f"Середній WAPE по 4 фолдах: {mean_wape:.4f}")

    # Бізнес-перевірка якості моделі
    if mean_wape > 0.25:
        raise ValueError(f"Якість моделі незадовільна! Mean WAPE: {mean_wape:.4f} > 0.25")

with DAG(
    dag_id="fmcg_demand_forecast_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["fmcg", "sagemaker", "lgbm"],
) as dag:

    # 1. Створення паралельних задач навчання на бектесті
    training_tasks = []
    cutoffs = [88, 92, 96, 100]

    for cutoff in cutoffs:
        task = DockerOperator(
            task_id=f"train_fold_cutoff_{cutoff}",
            image=ECR_IMAGE,
            api_version="auto",
            auto_remove=True,
            command=[
                "src/train.py",
                "--cutoff-week", str(cutoff),
                "--horizon", "4",
                "--data-path", "/opt/ml/input/data/training/dataset_104w_50s.parquet",
                "--output-folder", f"/opt/ml/model/fold_{cutoff}",
            ],
            docker_url="unix://var/run/docker.sock",
            network_mode="bridge",
            mounts=[
                {"source": f"{PROJECT_DIR}/dataset_104w_50s.parquet", "target": "/opt/ml/input/data/training/dataset_104w_50s.parquet", "type": "bind"},
                {"source": f"{PROJECT_DIR}/model_output", "target": "/opt/ml/model", "type": "bind"},
            ],
            environment={"PYTHONUNBUFFERED": "1"},
        )
        training_tasks.append(task)

    # 2. Агрегація метрик бектесту
    evaluate_task = PythonOperator(
        task_id="evaluate_backtest_metrics",
        python_callable=aggregate_metrics_fn,
    )

    # 3. Генерація бізнес-прогнозу за допомогою моделі останнього фолду (cutoff=100)
    predict_task = DockerOperator(
        task_id="generate_final_forecast",
        image=ECR_IMAGE,
        api_version="auto",
        auto_remove=True,
        command=[
            "src/predict.py",
            "--data-path", "/opt/ml/input/data/training/dataset_104w_50s.parquet",
            "--model-dir", "/opt/ml/model/fold_100",
            "--horizon", "4",
            "--output-path", "/opt/ml/model/forecast_h4.parquet",
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            {"source": f"{PROJECT_DIR}/dataset_104w_50s.parquet", "target": "/opt/ml/input/data/training/dataset_104w_50s.parquet", "type": "bind"},
            {"source": f"{PROJECT_DIR}/model_output", "target": "/opt/ml/model", "type": "bind"},
        ],
        environment={"PYTHONUNBUFFERED": "1"},
    )

    # 4. Вивантаження фінального прогнозу в Amazon S3
    upload_to_s3_task = LocalFilesystemToS3Operator(
        task_id="upload_forecast_to_s3",
        filename=f"{PROJECT_DIR}/model_output/forecast_h4.parquet",
        dest_key="forecasts/forecast_h4.parquet",
        dest_bucket=S3_BUCKET,
        replace=True,
    )

    # Зв'язування пайплайну: 4 фолди паралельно -> оцінка -> інференс -> завантаження в S3
    training_tasks >> evaluate_task >> predict_task >> upload_to_s3_task