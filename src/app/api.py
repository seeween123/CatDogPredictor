import io
import logging
import os
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("cats-dogs-api")

# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "../../serving/model/cats_dogs_cnn.keras")
IMG_SIZE = (224, 224)

# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------
app = FastAPI(
    title="Cats vs Dogs Inference Service",
    description="FastAPI inference service for the trained Cats vs Dogs CNN.",
    version="1.0.0",
)

# ---------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------
request_count = 0
successful_requests = 0
failed_requests = 0
total_latency = 0.0

# The service does not log sensitive information such as raw image files.

# Load the model once when the service starts.
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. "
        "Set MODEL_PATH or place cats_dogs_cnn.keras in ./model."
    )

model = load_model(MODEL_PATH)
logger.info("Model loaded successfully")


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------
def predict_image(image: Image.Image) -> dict:

    global request_count
    global successful_requests
    global failed_requests
    global total_latency

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    request_count += 1

    logger.info("Prediction request started | request_id=%s", request_id)

    """Preprocess an image and return class probabilities and label."""
    image = image.convert("RGB").resize(IMG_SIZE)

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Binary sigmoid output: probability of Dog.
    dog_probability = float(model.predict(image_array, verbose=0)[0][0])
    dog_probability = float(np.clip(dog_probability, 0.0, 1.0))
    cat_probability = 1.0 - dog_probability

    if dog_probability >= 0.5:
        label = "Dog"
        confidence = dog_probability
    else:
        label = "Cat"
        confidence = cat_probability

    latency = time.perf_counter() - start_time
    successful_requests += 1

    total_latency += latency

    logger.info(
        "Prediction completed | "
        "request_id=%s | "
        "prediction=%s | "
        "confidence=%.4f | "
        "latency_ms=%.2f",
        request_id,
        label,
        confidence,
        latency * 1000,
    )

    return {
        "label": label,
        "confidence": round(confidence, 6),
        "probabilities": {
            "Cat": round(cat_probability, 6),
            "Dog": round(dog_probability, 6),
        },
    }


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------


@app.get("/health")
def health_check():
    """Health/readiness endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict Cat/Dog from an uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = predict_image(image)

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            **result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {exc}",
        ) from exc


# ---------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------


@app.get("/metrics")
def metrics():

    average_latency = total_latency / request_count if request_count > 0 else 0

    return {
        "request_count": request_count,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "average_latency_ms": round(average_latency * 1000, 2),
    }
