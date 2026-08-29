import os

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def check_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        return f"API Status: {data['status']}\nModel loaded: {data['model_loaded']}"
    except Exception as exc:
        return f"API unavailable: {exc}"


def predict(image):
    if image is None:
        return "Please upload an image.", None

    try:
        # Gradio supplies a NumPy array when type='numpy'.
        from PIL import Image
        import io

        pil_image = Image.fromarray(image.astype("uint8"))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        response = requests.post(
            f"{API_URL}/predict",
            files={"file": ("image.png", buffer, "image/png")},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        result = (
            f"Prediction: {data['label']}\n"
            f"Confidence: {data['confidence'] * 100:.2f}%\n\n"
            f"Cat: {data['probabilities']['Cat'] * 100:.2f}%\n"
            f"Dog: {data['probabilities']['Dog'] * 100:.2f}%"
        )

        return result, data["probabilities"]

    except requests.RequestException as exc:
        return f"Prediction API error: {exc}", None
    except Exception as exc:
        return f"Error: {exc}", None

def get_metrics():
    try:
        response = requests.get(
            f"{API_URL}/metrics",
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        return (
            data.get("request_count", 0),
            data.get("successful_requests", 0),
            data.get("failed_requests", 0),
            data.get("average_latency_ms", 0)
        )

    except requests.exceptions.RequestException as e:
        return (
            "Error",
            "Error",
            "Error",
            str(e)
        )
    
with gr.Blocks(title="Cats vs Dogs Classifier") as demo:
    gr.Markdown(
        """
        # 🐱🐶 Cats vs Dogs Classifier

        Upload an image and the application will call the **FastAPI
        inference service** to classify it as Cat or Dog.
        """
    )

    health_output = gr.Textbox(
        label="API Health",
        value="Click 'Check API Health' to test the service.",
    )

    health_button = gr.Button("Check API Health")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="numpy",
                label="Upload Cat/Dog Image",
            )
            predict_button = gr.Button("Predict", variant="primary")

        with gr.Column():
            prediction_output = gr.Textbox(
                label="Prediction",
                lines=5,
            )
            probability_output = gr.Label(
                label="Class Probabilities",
            )

    health_button.click(
        fn=check_health,
        inputs=None,
        outputs=health_output,
    )

    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=[prediction_output, probability_output],
    )

    with gr.Row():

        request_count = gr.Number(
            label="Total Requests",
            value=0,
            interactive=False
        )

        successful_requests = gr.Number(
            label="Successful Requests",
            value=0,
            interactive=False
        )

        failed_requests = gr.Number(
            label="Failed Requests",
            value=0,
            interactive=False
        )

        average_latency = gr.Number(
            label="Average Latency (ms)",
            value=0,
            interactive=False
        )

    refresh_button = gr.Button(
        "Refresh Metrics"
    )

    refresh_button.click(
        fn=get_metrics,
        outputs=[
            request_count,
            successful_requests,
            failed_requests,
            average_latency
        ]
    )

    # Automatically refresh every 5 seconds
    timer = gr.Timer(
        value=5
    )

    timer.tick(
        fn=get_metrics,
        outputs=[
            request_count,
            successful_requests,
            failed_requests,
            average_latency
        ]
    )

if __name__ == "__main__":
    demo.launch()
