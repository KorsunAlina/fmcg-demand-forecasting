from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

CUTOFFS = [88, 92, 96, 100]
HORIZON = 4

# Точний шлях до папки проєкту на вашому Windows комп'ютері
HOST_PROJECT_PATH = r"C:\Users\akors\OneDrive\Desktop\evowill\test_project"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 0,
}

dag = DAG(
    "fmcg_demand_forecasting_pipeline",
    default_args=default_args,
    description="Паралельний бектест 4 фолдів, Quality Gate та фінальний прогноз",
    schedule_interval=None,
    catchup=False,
)

# Монтування каталогу хоста Windows усередину Docker-контейнера задачі
project_mount = Mount(
    target="/opt/program/data",
    source=HOST_PROJECT_PATH,
    type="bind",
)

backtest_tasks = []

# 1. 4 паралельні задачі бектесту
for cutoff in CUTOFFS:
    task = DockerOperator(
        task_id=f"train_backtest_cutoff_{cutoff}",
        image="fmcg-forecasting:latest",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,  # КРИТИЧНО: вимикає автоматичне монтування неіснуючої /tmp теки
        mounts=[project_mount],
        # Замість http://host.docker.internal:5000
        environment={
            "MLFLOW_TRACKING_URI": "file:/opt/program/data/mlruns",
            "MLFLOW_ALLOW_FILE_STORE": "true",
        },
        command=[
            "-m", "src.train",
            "--data-path", "/opt/program/data/dataset_104w_50s.parquet",
            "--cutoff-week", str(cutoff),
            "--horizon", str(HORIZON),
            "--output-folder", "/opt/program/data/artifacts",
        ],
        dag=dag,
    )
    backtest_tasks.append(task)


# 2. Перевірка результатів (Quality Gate)
def evaluate_and_check_quality(**context):
    print("Усі 4 бектест-фолди завершилися успішно.")
    features_file = "/opt/project/artifacts/features.json"
    model_file = "/opt/project/artifacts/model.pkl"

    if not os.path.exists(features_file) or not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Артефакти не знайдено! Перевірте наявність {features_file} та {model_file}"
        )

    print("Quality Gate пройдено успішно!")
    return True


quality_gate_task = PythonOperator(
    task_id="quality_gate",
    python_callable=evaluate_and_check_quality,
    dag=dag,
)

# 3. Фінальний прогноз
predict_task = DockerOperator(
    task_id="generate_final_forecast",
    image="fmcg-forecasting:latest",
    api_version="auto",
    auto_remove=True,
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    mount_tmp_dir=False,  # КРИТИЧНО: вимикає автоматичне монтування неіснуючої /tmp теки
    mounts=[project_mount],
    command=[
        "-m", "src.predict",
        "--data-path", "/opt/program/data/dataset_104w_50s.parquet",
        "--model-dir", "/opt/program/data/artifacts",
        "--horizon", str(HORIZON),
        "--output-path", "/opt/program/data/forecasts.parquet",
    ],
    dag=dag,
)

# Порядок виконання
backtest_tasks >> quality_gate_task >> predict_task