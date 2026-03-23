# Customer Churn Prediction — Tabular Classification

## Overview
A machine learning pipeline predicting which customers are likely to churn next month. Compares multiple classification models with rigorous experiment tracking in MLflow and automated retraining via Airflow.

## Architecture
- **Data**: Telco/SaaS customer data (tenure, plan, usage, support tickets, payment history)
- **Feature Engineering**: Behavioral aggregations, recency-frequency-monetary features, interaction terms
- **Models**: Logistic Regression (baseline) → XGBoost → LightGBM with Optuna hyperparameter search
- **Evaluation**: AUC-ROC, Precision-Recall curves, threshold tuning optimized for recall (cost of losing a customer > cost of false alarm)
- **Tracking**: MLflow experiment logging — parameters, metrics, confusion matrices, feature importance plots
- **Deployment**: Best model registered in MLflow Model Registry, Airflow DAG for monthly retrain

## Tech Stack
- Python 3.11+, Pandas, scikit-learn
- XGBoost, LightGBM, Optuna
- MLflow (tracking + model registry)
- Apache Airflow 2.x
- Docker & Docker Compose

## What This Demonstrates
- End-to-end ML pipeline from data prep to model registry
- Model comparison and selection methodology
- Threshold tuning for business-aware metrics
- Feature importance and model interpretability
- MLflow experiment tracking best practices
- Automated retraining pipelines

## Status
🚧 In Development

## Project Structure
├── dags/
│   └── churn_retraining_dag.py
├── src/
│   ├── features/
│   ├── training/
│   └── evaluation/
├── notebooks/
├── tests/
├── mlflow/
├── docker-compose.yml
└── README.md
