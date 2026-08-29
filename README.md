## Heart Disease Prediction – End-to-End Machine Learning Project
### Purpose

Build and deploy an end-to-end machine learning system that predicts
the presence of heart disease using the UCI Heart Disease dataset. An end-to-end Machine Learning and MLOps project for predicting the likelihood of heart disease using clinical patient data from the UCI Heart Disease dataset. The project demonstrates the complete machine learning lifecycle, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter optimization with Optuna, experiment tracking with MLflow, model serving using FastAPI, containerization with Docker, and CI/CD automation using GitHub Actions.

---

### Video Walkthrough
Please refer to Youtube video: [link](https://youtu.be/bP0nUIUyE4U)

---

### Problem solved & benefits

Early identification of patients at risk of cardiovascular disease.

Clinical decision support.

REST API for inference.

MLflow experiment tracking.

Docker deployment - Docker + MiniKube

FastAPI service.

Interactive Gradio dashboard.

CI/CD using GitHub Actions.


### Details

- Data & Modeling: Feature engineering + XGBoost classifier; experiments logged to MLflow.
- Model tracking: Runs, metrics, and the serialized model logged under a named MLflow experiment.
- Inference service: FastAPI app exposing /predict (POST)
- Web UI: Gradio interface mounted at /ui for quick, shareable manual testing.
- Containerization: Docker image with uvicorn entrypoint (src.app.main:app) listening on port 8000.
- CI/CD: GitHub Actions builds the image and pushes to Docker Hub. Local or AWS ECS 
- Orchestration: MiniKube
- Networking: Local / AWS
- Security: Local / AWS
- Observability: Local / AWS

## Project Overview

This project was developed to demonstrate production-ready machine learning practices by combining modern MLOps tools with a robust prediction pipeline.

The workflow includes:

- Data acquisition and validation
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training
- Hyperparameter optimization using Optuna
- Experiment tracking with MLflow
- Model Registry
- Model serving with FastAPI
- Docker containerization
- MiniKube Orchestration
- GitHub Actions CI/CD
- Lint and Unit testing

---

## Features

- End-to-end ML pipeline
- Automated data validation
- Comprehensive EDA
- Feature engineering pipeline
- Multiple machine learning models
- Optuna hyperparameter optimization
- MLflow experiment tracking and model registry
- REST API using FastAPI
- Docker and MiniKube support
- GitHub Actions CI pipeline
- Unit tests with pytest

---

## Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| API | FastAPI |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Containerization | Docker |
| Container Orchestration | Minikube |

---

# Project Structure

```
HeartDiseasePredictor/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   └── EDA.ipynb
│
├── serving/
│   └── model/
│
├── scripts/
│
├── src/
│   ├── app/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── utils/
│   └── serving/
│
├── tests/
│
├── app.py
├── main.py
├── run_pipeline.py
├── Dockerfile
├── requirements.txt
├── deployment.yaml
├── service.yaml
├── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/seeween123/HeartDiseasePredictor.git

cd HeartDiseasePredictor
```

---

## Create Virtual Environment
### Environment Setup

Create and activate a Python virtual environment:

### Windows

```powershell
# Create virtual environment
Python3.11 -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

> **Note:** Ensure that `uv` is installed before running the last command. If not, install it using:
>
> ```powershell
> pip install uv
> ```

### Linux/macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

# Running the Project

Run the complete pipeline

```bash
python run_pipeline.py
```

or

```bash
python src/main.py
```

---

# Running the API

```bash
uvicorn app:app --reload
```

Open

```
http://localhost:8000/docs
```

to access the interactive Swagger UI.

---

# Running Tests

```bash
pytest tests/
```

---

# Exploratory Data Analysis

The EDA notebook performs:

- Missing value analysis
- Duplicate detection
- Data validation
- Statistical summaries
- Distribution analysis
- Boxplots
- Correlation heatmap
- Multicollinearity analysis (VIF)
- Feature importance exploration

Outputs generated include:

- Histograms
- Boxplots
- Correlation Matrix
- VIF Report
- Data Summary

---

# Feature Engineering

The preprocessing pipeline includes:

- Missing value handling
- One-hot encoding
- Feature scaling
- Train/Test split
- Feature selection
- Processed dataset generation

---

# Machine Learning Models

The following algorithms were evaluated:

- Random Forest
- XGBoost
- LightGBM

Model performance was evaluated using:

- Recall
- F1 Score
- Accuracy
- Precision
- ROC-AUC

For this medical classification problem, Recall was prioritized to reduce false negatives.

---

# Hyperparameter Optimization

Hyperparameter tuning is performed using **Optuna**.

Optuna employs Bayesian optimization using the Tree-structured Parzen Estimator (TPE) sampler to efficiently search for optimal model parameters.

Parameters optimized include:

### XGBoost

- learning_rate
- max_depth
- n_estimators
- min_child_weight
- gamma
- subsample
- colsample_bytree
- reg_alpha
- reg_lambda

Benefits of Optuna:

- Faster than Grid Search
- Bayesian optimization
- Automatic pruning of poor trials
- Reduced computation time
- Better model performance

---

# MLflow Experiment Tracking

MLflow is used to record every experiment.

Each run stores:

- Hyperparameters
- Metrics
- Artifacts
- Trained models
- Feature importance
- Configuration files
- Evaluation reports

The best model is registered in the MLflow Model Registry before deployment.

---

# Model Deployment

The selected production model is exported into the serving directory.

FastAPI loads the production model and exposes REST endpoints for prediction.

Workflow:

```
Dataset
     │
     ▼
EDA
     │
     ▼
Feature Engineering
     │
     ▼
Model Training
     │
     ▼
Optuna Optimization
     │
     ▼
MLflow Tracking
     │
     ▼
Model Registry
     │
     ▼
Serving Model
     │
     ▼
FastAPI API
```


---

# Docker

Build Docker image

```bash
docker build -t heart-disease-predictor .
```

Run container

```bash
docker run -p 8000:8000 heart-disease-predictor
```

---

# Container Orchestration

This project uses Minikube for creating a virtual cluster and orchestrate the docker contatiner. Deployment.yaml and Service.yaml are used for Minikube configuration. Stantard commands are used to load/unload the container+pod on the target vitual nodes. Please use the following set of commands:


- minikube start
- minikube docker-env | Invoke-Expression
- docker build -t cats-dogs-api:latest .
- kubectl create deployment cats-dogs-deployment --image=cats-dogs-api:latest --port=8000
- kubectl expose deployment cats-dogs-deployment --type=NodePort --port=8000
- minikube service cats-dogs-deployment
- kubectl get pods
- minikube service cats-dogs-deployment --url
- kubectl port-forward service/cats-dogs-deployment 8000:8000


Once the service is up, use the following commands to test the deployment:
- curl.exe "$env:API_URL/health"
- curl.exe "$env:API_URL/metrics"
- curl.exe "$env:API_URL/predict" -F "file=@TestImage.jpg"
- curl.exe "$env:API_URL/metrics"

# CI/CD

GitHub Actions automates the following tasks:

- Checkout repository
- Install dependencies
- Lint source code. Use ruff to find and fix code issues.
- Execute unit tests
- Build Docker image
- Validate application

The pipeline executes automatically for:

- Push to `main`
- Pull Requests

---

# Architecture

```
                    Raw Dataset
                         │
                         ▼
                 Data Validation
                         │
                         ▼
               Exploratory Data Analysis
                         │
                         ▼
                Feature Engineering
                         │
                         ▼
                Train / Test Split
                         │
                         ▼
       ┌───────────────┬───────────────┬───────────────┐
       ▼               ▼               ▼
 Logistic Regression Random Forest  XGBoost/LightGBM
       │               │               │
       └───────────────┴───────────────┘
                         │
                         ▼
                 Optuna Optimization
                         │
                         ▼
                 MLflow Experiment Tracking
                         │
                         ▼
                  MLflow Model Registry
                         │
                         ▼
                 Production Model Export
                         │
                         ▼
                    FastAPI Service
                         │
                         ▼
                     Client Request
```

---

# Repository

https://github.com/seeween123/HeartDiseasePredictor

---

# Future Updates to include

- Model monitoring
- Drift detection
- Automated retraining
- Kubernetes deployment
- Cloud deployment (AWS/Azure/GCP)
- Feature Store integration
- Explainable AI (SHAP/LIME)

---

# Project Screenshots

Attached are screenshots demonstrating the project's architecture, experiment tracking, CI/CD pipeline, model serving, and deployment workflow.

---

<img src="docs/images/Screenshot 2026-07-12 091109.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 091206.png" width="900">

---

<img src="docs/images/CorrelationMatrix" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 091228.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 091305.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 091345.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 091431.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 094116.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 094131.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 094148.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 094210.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 094305.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 094414.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 095639.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 095652.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 115431.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 111248.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 114926.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 125638.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 125829.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 135437.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 140032.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 141309.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 142953.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 143709.png" width="900">

---

<img src="docs/images/Screenshot 2026-07-12 150554.png" width="900">

---