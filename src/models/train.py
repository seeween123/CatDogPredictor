import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.models import infer_signature
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def train_model(
    train_generator,
    val_generator,
    test_generator,
    model_path="../serving/model/cats_dogs_cnn.keras",
):
    """
    Trains a CNN model and logs with MLflow.

    Args:
        train_generator: Training data generator.
        val_generator: Validation data generator.
        test_generator: Test data generator.
    """

    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("Cats_Dogs_CNN")

    print("MLflow Tracking URI:", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name="CNN_Training"):
        # Log model/training parameters
        mlflow.log_params(
            {
                "img_size": str(IMG_SIZE),
                "batch_size": BATCH_SIZE,
                "epochs": 5,
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "augmentation": True,
                "rotation_range": 20,
                "width_shift_range": 0.2,
                "height_shift_range": 0.2,
                "shear_range": 0.2,
                "zoom_range": 0.2,
                "horizontal_flip": True,
            }
        )

        history = model.fit(train_generator, validation_data=val_generator, epochs=5)

        # Log metrics for every epoch
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(
            zip(
                history.history["loss"],
                history.history["accuracy"],
                history.history["val_loss"],
                history.history["val_accuracy"],
            )
        ):
            mlflow.log_metrics(
                {
                    "train_loss": loss,
                    "train_accuracy": acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                step=epoch,
            )

        # Evaluate on the held-out test set and log final metrics
        test_loss, test_acc = model.evaluate(test_generator, verbose=1)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

        # Save and log the trained model as an MLflow artifact
        model.save(model_path)
        mlflow.log_artifact(model_path)

        # Log the TensorFlow/Keras model in MLflow's model format
        signature = infer_signature(
            np.zeros((1, 224, 224, 3), dtype=np.float32),
            model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0),
        )
        mlflow.tensorflow.log_model(model, name="cnn_model", signature=signature)

        print("Test Accuracy:", test_acc)
        print("MLflow Run ID:", mlflow.active_run().info.run_id)
