# FMCG Demand Forecasting: Production ML Pipeline

[![CI/CD Pipeline](https://github.com/KorsunAlina/fmcg-demand-forecasting/actions/workflows/deploy.yml/badge.svg)](https://github.com/KorsunAlina/fmcg-demand-forecasting/actions/workflows/deploy.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![AWS ECR](https://img.shields.io/badge/AWS-ECR%20%7C%20S3-FF9900.svg)](https://aws.amazon.com/)
[![Airflow](https://img.shields.io/badge/Airflow-2.9-017CEE.svg)](https://airflow.apache.org/)

Production-grade Machine Learning pipeline designed to forecast 4-week ahead sales for Fast-Moving Consumer Goods (FMCG).

The project covers the complete ML engineering lifecycle: recursive time-series feature engineering with data leakage prevention, LightGBM training, containerization under AWS SageMaker contracts, Apache Airflow orchestration, and automated deployment via GitHub Actions CI/CD.

---

## 1. Business Context and Problem Statement

- Objective: Predict weekly SKU-level sales across retail stores to avoid stockouts while minimizing holding costs.
- Planning Horizon: 4 weeks ahead (h = 4).
- ML Task: Autoregressive multi-step time-series forecasting using LightGBM.
- Key Challenge: Preventing target leakage across future recursive steps by strictly recomputing rolling statistics, calendar indicators, and lag dynamics within expanding training windows.

---

## 2. Model Performance and Backtest Results

Validation was conducted via expanding window backtesting across historical splits (cutoff_week: 88, 92, 96, 100).

- Model WAPE: 20.95%
- Baseline WAPE: 25.91%
- Error Reduction: +4.96 pp (~19% relative improvement)
- MASE: 0.8085 (outperforms baseline benchmark < 1.0)

---

## 3. Architecture and Design Decisions

- Cloud Storage: Amazon S3 (s3://fmcg-demand-forecast-alina-2026) for dataset inputs and forecast outputs.
- Container Registry: Amazon ECR (eu-north-1) for production Docker images.
- SageMaker File Contract: Docker containers adhere to SageMaker execution standards (/opt/ml/input/data/training and /opt/ml/model).
- Workflow Orchestration: Apache Airflow DAG executing parallel backtest folds, metric validation (WAPE < 25%), inference, and S3 upload.
- CI/CD Automation: GitHub Actions workflow triggers flake8 checks and pushes updated Docker images to Amazon ECR on push to main branch.

---

## 4. Repository Structure

- .github/workflows/deploy.yml: CI/CD workflow for automated ECR deployment.
- dags/fmcg_pipeline_dag.py: Apache Airflow pipeline definition.
- src/data_prep.py: Feature engineering pipeline (lags, rolling averages, date features).
- src/train.py: Model training, backtest evaluation, and metric calculation.
- src/predict.py: Recursive 4-week demand forecasting script.
- Dockerfile: Container configuration based on Python 3.12-slim.
- requirements.txt: Python dependencies.
- .gitignore: Excluded local files and environments.
- README.md: Project documentation.

---

## 5. Execution Guide

Prerequisites:
- Docker Desktop
- AWS CLI configured
- Python 3.12+

1. Local Environment Setup:
git clone https://github.com/KorsunAlina/fmcg-demand-forecasting.git
cd fmcg-demand-forecasting
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

2. Run Training via Docker:
docker run --rm -e PYTHONUNBUFFERED=1 -v "$($PWD.Path)\dataset_104w_50s.parquet:/opt/ml/input/data/training/dataset_104w_50s.parquet" -v "$($PWD.Path)\model_output:/opt/ml/model" fmcg-forecasting:latest src/train.py --cutoff-week 100 --horizon 4 --data-path /opt/ml/input/data/training/dataset_104w_50s.parquet --output-folder /opt/ml/model

3. Run Inference via Docker:
docker run --rm -e PYTHONUNBUFFERED=1 -v "$($PWD.Path)\dataset_104w_50s.parquet:/opt/ml/input/data/training/dataset_104w_50s.parquet" -v "$($PWD.Path)\model_output:/opt/ml/model" fmcg-forecasting:latest src/predict.py --data-path /opt/ml/input/data/training/dataset_104w_50s.parquet --model-dir /opt/ml/model --horizon 4 --output-path /opt/ml/model/forecast_h4.parquet

---

## 6. GitHub Actions Secrets

Configured under Repository Settings -> Secrets and variables -> Actions:
- AWS_ACCESS_KEY_ID: AWS IAM user access key.
- AWS_SECRET_ACCESS_KEY: AWS IAM user secret key.
- AWS Region: eu-north-1.