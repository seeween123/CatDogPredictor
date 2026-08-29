#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering
"""

import os
import sys
import argparse
import mlflow
import mlflow.sklearn


from src.models.tune import tune_model
from src.models.train import train_model
from models.predict import predict_model

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules - Core pipeline components
from src.data.make_dataset import load_data  # Data loading with error handling
from src.data.preprocessing import preprocess_data  # Basic data cleaning
from src.features.build_features import (
    build_features,
)  # Feature engineering (CRITICAL for model performance)
from src.utils.validate_data import (
    validate_heart_disease_data,
)  # Data quality validation


def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.

    """

    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    # Configure MLflow to use local file-based tracking (not a tracking server)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = (
        args.mlflow_uri or f"file://{project_root}/mlruns"
    )  # Local file-based tracking
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  # Creates experiment if doesn't exist

    # Start MLflow run - all subsequent logging will be tracked under this run
    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        # REQUIRED: These parameters are essential for model reproducibility
        mlflow.log_param("model", "xgboost")  # Model type for comparison
        mlflow.log_param(
            "threshold", args.threshold
        )  # Classification threshold (default: 0.35)
        mlflow.log_param("test_size", args.test_size)  # Train/test split ratio

        # === STAGE 1: Data Loading & Validation ===
        print("🔄 Loading data...")
        df = load_data(args.input)  # Load raw CSV data with error handling
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === CRITICAL: Data Quality Validation ===
        # This step is ESSENTIAL for production ML - validates data quality before training
        print("🔍 Validating data quality with Great Expectations...")
        is_valid, failed = validate_heart_disease_data(df)
        mlflow.log_metric(
            "data_quality_pass", int(is_valid)
        )  # Track data quality over time

        if not is_valid:
            # Log validation failures for debugging
            import json

            mlflow.log_text(
                json.dumps(failed, indent=2), artifact_file="failed_expectations.json"
            )
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed. Logged to MLflow.")

        # === STAGE 2: Data Preprocessing ===
        print("🔧 Preprocessing data...")
        df = preprocess_data(
            df
        )  # Basic cleaning (handle missing values, fix data types)

        # Save processed dataset for reproducibility and debugging
        processed_path = os.path.join(
            project_root, "data", "processed", "heart_disease_processed.csv"
        )
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"✅ Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === STAGE 3: Feature Engineering - CRITICAL for Model Performance ===
        print("🛠️  Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")

        # Apply feature engineering transformations
        df_enc = build_features(
            df, target_col=target
        )  # Binary encoding + one-hot encoding

        # IMPORTANT: Convert boolean columns to integers for XGBoost compatibility
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"✅ Feature engineering completed: {df_enc.shape[1]} features")

        # === CRITICAL: Save Feature Metadata for Serving Consistency ===
        # This ensures serving pipeline uses exact same features in exact same order
        import json
        import joblib

        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Get feature columns (exclude target)
        feature_cols = list(df_enc.drop(columns=[target]).columns)

        # Save locally for development serving
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        # Log to MLflow for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # ESSENTIAL: Save preprocessing artifacts for serving pipeline
        # These artifacts ensure training and serving use identical transformations
        preprocessing_artifact = {
            "feature_columns": feature_cols,  # Exact feature order
            "target": target,  # Target column name
        }
        joblib.dump(
            preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl")
        )
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"✅ Saved {len(feature_cols)} feature columns for serving consistency")

        # === STAGE 5: Hyperparameter Tuning ===
        print("🔍 Running Optuna hyperparameter tuning...")

        best_params = tune_model(df_enc, target_col=target)
        print("Best Parameters:", best_params)

        # === STAGE 4: Model Training ===
        print("🤖 Training model...")

        best_model = train_model(df=df_enc, target_col=target, best_params=best_params)
        # === STAGE 5: Model Evaluation ===
        print("📊 Evaluating model...")
        predict_model(best_model, df_enc, target)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run Heart Disease Predictor pipeline with XGBoost + MLflow"
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to CSV (e.g., data/raw/HeartDisease.csv)",
    )
    p.add_argument("--target", type=str, default="target")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Heart Disease Predictor")
    p.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="override MLflow tracking URI, else uses project_root/mlruns",
    )

    args = p.parse_args()
    main(args)

"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py \                                            
    --input data/raw/HeartDisease.csv \
    --target target

"""
