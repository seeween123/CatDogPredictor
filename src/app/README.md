# Cats vs Dogs Inference Service

This application wraps the trained `cats_dogs_cnn.keras` model in a FastAPI REST API and provides a simple Gradio UI.

## Start the FastAPI service

CD to this project directory:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### Health check

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

### Prediction

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

## Start the Gradio UI

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

## Model preprocessing

The service follows the training preprocessing:
- RGB image
- Resize to 224 x 224
- Normalize pixel values to [0, 1]
- Sigmoid output interpreted as Dog probability
- Cat probability = 1 - Dog probability
