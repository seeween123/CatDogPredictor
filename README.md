## Cat Dog Prediction for Pet Adoption Agency – End-to-End Machine Learning Project

### Repository

https://github.com/seeween123/CatDogPredictor

---

### Video Walkthrough
Please refer to Youtube video: [link](https://youtu.be/8_ZXIwL_tW8)

---

### Purpose

Design, develop, and deploy a scalable, reproducible machine learning
solution using modern MLOps best practices. The assignment emphasizes practical
automation, experiment tracking, CI/CD pipelines, containerization, cloud deployment,
and monitoring, mirroring real-world production scenarios.

#### Use case : Binary image classification (Cats vs Dogs) for a pet adoption platform.

#### Dataset : Cats and Dogs classification dataset
CATS and Dogs binary classification dataset from Kaggle
Pre-process to 224x224 RGB images for standard CNNs

Split into train/validation/test sets (e.g., 80%/10%/10%). Use data augmentation for better generalization

---

### Problem solved & benefits

Identification of Cat vs Dog in give set of images using CNN model.

REST API for inference.

MLflow experiment tracking.

Docker deployment - Docker + MiniKube

FastAPI service - Health, Predict and Metrics endpoints

Interactive Gradio UI.

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

Key Features includes:

- Data acquisition
- Data augmentation
- Train, Validation, Test split
- CNN Model training
- Experiment tracking with MLflow
- Git-LFS for model storage
- Git repository for code and scripts
- Model Registry
- Model serving with FastAPI
- Metrics and logging via
- Gradio UI application
- Post deployment Smoke test
- Post deploment Model Perf tracking
- Docker containerization
- MiniKube Orchestration
- GitHub Actions CI/CD
- PyLint, Ruff checking 
- Pytest Unit testing

---

## Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Tensorflow CNN |
| Experiment Tracking | MLflow |
| API | FastAPI |
| UI | Gradio |
| CI/CD | GitHub Actions |
| Code check | PyLint/Ruff |
| Testing | Pytest |
| Containerization | Docker |
| Container Orchestration | Minikube |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/seeween123/CatDogPredictor.git

cd CatDogPredictor
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

## Running the Project

Run the complete pipeline

```bash
python run_pipeline.py
```

## Cats vs Dogs Inference Service

This application wraps the trained `cats_dogs_cnn.keras` model in a FastAPI REST API and provides a simple Gradio UI.

### Start the FastAPI service

CD to this project directory:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### Endpoints

#### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "./model/cats_dogs_cnn.keras"
}
```

#### Prediction

```http
POST /predict
```

Upload an image as multipart form data using the field name `file`.

Example with curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@cat.jpg"
```

Example response:

```json
{
  "filename": "cat.jpg",
  "content_type": "image/jpeg",
  "label": "Cat",
  "confidence": 0.984321,
  "probabilities": {
    "Cat": 0.984321,
    "Dog": 0.015679
  }
}
```

### Start the Gradio UI

Keep the FastAPI server running, then open another terminal:

```bash
python ui.py
```

The Gradio UI will call:

```text
http://127.0.0.1:8000/predict
```

If the API is running on another address, set:

Windows PowerShell:

```powershell
$env:API_URL="http://127.0.0.1:8000"
python ui.py
```

### Model preprocessing

The service follows the training preprocessing:
- RGB image
- Resize to 224 x 224
- Normalize pixel values to [0, 1]
- Sigmoid output interpreted as Dog probability
- Cat probability = 1 - Dog probability


---

## Running Tests


### Pytest tests for the Cats vs Dogs CNN pipeline

These tests cover:

- `preprocessing.py`
  - image copying
  - class directory assignment
  - JPG filtering
  - 224x224 image generators
  - batch size and binary classification
  - test generator shuffle setting
- `train.py`
  - CNN layer structure
  - optimizer/loss/metrics
  - five training epochs
  - evaluation and MLflow logging
- `predict.py`
  - Dog prediction at probability >= 0.5
  - Cat prediction at probability < 0.5
  - image resizing and normalization
- `make_dataset.py`
  - current source-code validity issue
  - expected Kaggle dataset and ZIP names

### Run

From the directory containing the four source files:

```bash
pip install pytest
pytest -v tests
```

The preprocessing and training tests require the project's TensorFlow and
MLflow dependencies.

## Machine Learning Models

The following algorithms were evaluated:

- CNN

Model performance was evaluated using:

- Recall
- F1 Score
- Accuracy
- Precision
- ROC-AUC

---

## MLflow Experiment Tracking

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

## Model Deployment

The selected production model is exported into the serving directory.

FastAPI loads the production model and exposes REST endpoints for prediction.

Workflow:

```
Data Acquisition
     │
     ▼
Data Augmentation  
     │
     ▼
Model Training
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

## Docker

Build Docker image

```bash
docker build -t cat-dog-api .
```

Run container

```bash
docker run -p 8000:8000 cat-dog-api
```

---

## Container Orchestration

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

## CI/CD

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
                 Data Augmentation
                         │
                         ▼
                Train / Test Split
                         │
                         ▼
                     CNN model
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

<img src="docs/images/Screenshot 2026-07-31 144801.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-26 061706.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-26 061728.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-26 061812.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-26 061824.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-26 061944.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-26 063944.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-28 030927.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-28 031037.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-28 031056.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-28 031140.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-28 034029.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-28 041734.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-28 043254.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-28 043312.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-29 153056.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-29 170206.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-29 212929.png" width="900">

---

<img src="docs/images/Screenshot 2026-08-29 213431.png" width="900">

---
