import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img

MODEL_PATH = "../serving/model/cats_dogs_cnn.keras"


def predict_model(image_path: str) -> tuple[str, float]:
    """Predict whether an image contains a cat or a dog."""
    IMG_SIZE = (224, 224)

    model = load_model(MODEL_PATH)
    image = load_img(image_path, target_size=IMG_SIZE)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    dog_probability = float(
        model.predict(image_array, verbose=0)[0][0]
    )

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

    return label, confidence


if __name__ == "__main__":
    image_path = r"../dataset/Processed/cats_dogs_split/test/cat/42.jpg"
    predict_model(image_path)
