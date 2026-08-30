import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img

MODEL_PATH = "./model/cats_dogs_cnn.keras"


def predict_model(image_path: str) -> tuple[str, float]:
    """
    Evaluates an XGBoost model on test data.

    Args:
        model: Trained model.
        df: Test dataset.
        target_col: Name of the target column.
    """
    IMG_SIZE = (224, 224)

    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    image = load_img(image_path, target_size=IMG_SIZE)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Sigmoid output represents the probability of the dog class.
    dog_probability = float(model.predict(image_array, verbose=0)[0][0])

    if dog_probability >= 0.5:
        label = "Dog"
        confidence = dog_probability
    else:
        label = "Cat"
        confidence = 1.0 - dog_probability

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Prediction: {label} ({confidence * 100:.2f}%)")
    plt.show()

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Dog probability: {dog_probability * 100:.2f}%")

    return label, confidence


# Example using an image from the test dataset:
image_path = r"../dataset/Processed/cats_dogs_split/test/cat/42.jpg"

predict_cat_or_dog(image_path)
