import os

import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

API_URL = "http://localhost:8000/predict"

DATASET_DIR = "post_deployment_data"


results = []


# ---------------------------------------------------------
# Send images to deployed API
# ---------------------------------------------------------

for label in ["Cat", "Dog"]:
    folder = os.path.join(DATASET_DIR, label)

    if not os.path.exists(folder):
        continue

    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder, filename)

        try:
            with open(image_path, "rb") as image:
                response = requests.post(
                    API_URL, files={"file": (filename, image, "image/jpeg")}, timeout=30
                )

            if response.status_code != 200:
                print(f"❌ Prediction test failed: HTTP {response.status_code}")
                print(response.text)

            result = response.json()

            # print(result)

            response.raise_for_status()

            result = response.json()

            predicted_label = result["label"]

            results.append(
                {
                    "image": filename,
                    "true_label": label,
                    "predicted_label": predicted_label,
                    "confidence": result["confidence"],
                    # "latency_ms": result["latency_ms"]
                }
            )

            print(f"{filename}: true={label}, predicted={predicted_label}")

        except Exception as e:
            print(f"Failed: {filename} -> {e}")


# ---------------------------------------------------------
# Create results DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(results)

if len(df) == 0:
    raise RuntimeError("No successful predictions collected.")


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

y_true = df["true_label"]
y_pred = df["predicted_label"]


accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred, pos_label="Dog", zero_division=0)

recall = recall_score(y_true, y_pred, pos_label="Dog", zero_division=0)

f1 = f1_score(y_true, y_pred, pos_label="Dog", zero_division=0)


# ---------------------------------------------------------
# Display metrics
# ---------------------------------------------------------

print("\n================================")
print("Post-Deployment Model Evaluation")
print("================================")

print(f"Number of requests: {len(df)}")

print(f"Accuracy:  {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1 Score:  {f1:.4f}")


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

print("\nConfusion Matrix:")

print(confusion_matrix(y_true, y_pred, labels=["Cat", "Dog"]))


# ---------------------------------------------------------
# Detailed classification report
# ---------------------------------------------------------

print("\nClassification Report:")

print(classification_report(y_true, y_pred, labels=["Cat", "Dog"], zero_division=0))


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

df.to_csv("post_deployment_results.csv", index=False)

print("\nResults saved to post_deployment_results.csv")
