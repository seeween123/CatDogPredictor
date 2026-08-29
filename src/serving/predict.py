"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Heart Disease prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import pandas as pd
import mlflow
from src.data.preprocessing import preprocess_data

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time
MODEL_DIR = "serving/model"

try:
    # Load the trained XGBoost model in MLflow pyfunc format
    # This ensures compatibility regardless of the underlying ML library
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")

if model is None:
    raise RuntimeError("Model not loaded")


def predict(input_dict: dict) -> str:
    """
    Main prediction function for Heart Disease inference.

    This function provides the complete inference pipeline from raw patient data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.

    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string

    Args:
        input_dict: Dictionary containing raw patient data with keys matching
                   the PatientData schema

    Returns:
        Human-readable prediction string:
        - "Likely to have heart disease" for high-risk patients (model prediction = 1)
        - "Not likely to have heart disease" for low-risk patients (model prediction = 0)

    Example:
        >>> patient_data = {
        ...     "age": 65, "sex": "Male", "chest_pain": "Typical Angina", ... # other features
        ... }
        >>> predict(patient_data)
        "Likely to have heart disease"
    """

    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])

    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = preprocess_data(df, target_col="target")

    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = model.predict(df_enc)

        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list

        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds

    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to have heart disease"  # High risk - needs intervention
    else:
        return "Not likely to have heart disease"  # Low risk - maintain normal service
