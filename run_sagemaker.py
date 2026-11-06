import sys
import boto3
import sagemaker
from sagemaker.estimator import Estimator

print("1. Ініціалізація сесії AWS та LocalSession...")
sys.stdout.flush()

# Створюємо базову boto3-сесію
boto_session = boto3.Session(region_name="eu-north-1")

# Створюємо локальну сесію SageMaker
session = sagemaker.LocalSession(boto_session=boto_session)
session.config = {"local": {"local_code": True}}

# Дані вашого середовища
account_id = "209115427590"
region = "eu-north-1"
role_arn = f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole-20261102T164417"
image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/fmcg-forecasting:latest"

print("2. Конфігурація Estimator (Local Mode)...")
estimator = Estimator(
    image_uri=image_uri,
    role=role_arn,
    instance_count=1,
    instance_type="local",  # Запуск локально через Docker Desktop без квот AWS
    output_path="s3://fmcg-demand-forecast-alina-2026/artifacts/",
    base_job_name="fmcg-backtest",
    sagemaker_session=session,
)

# Аргументи командного рядка для train.py
estimator.set_hyperparameters(
    cutoff_week=100,
    horizon=4,
    data_path="/opt/ml/input/data/training/dataset_104w_50s.parquet",
    output_folder="/opt/ml/model",
)

print("3. Запуск контейнера та навчання...")
sys.stdout.flush()

estimator.fit({
    "training": "s3://fmcg-demand-forecast-alina-2026/data/"
})

print("4. Навчання в SageMaker Local Mode успішно завершено!")