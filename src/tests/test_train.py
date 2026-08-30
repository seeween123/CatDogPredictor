
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from conftest import load_module

train = load_module("train(3).py", "train")


def test_train_model_builds_expected_cnn(tmp_path):
    class DummyHistory:
        history = {
            "loss": [0.8],
            "accuracy": [0.7],
            "val_loss": [0.75],
            "val_accuracy": [0.72],
        }

    fake_model = MagicMock()
    fake_model.fit.return_value = DummyHistory()
    fake_model.evaluate.return_value = (0.6, 0.8)
    fake_model.predict.return_value = np.array([[0.75]], dtype=np.float32)

    fake_run = MagicMock()
    fake_run.__enter__.return_value = fake_run
    fake_run.info.run_id = "test-run-id"

    fake_mlflow = MagicMock()
    fake_mlflow.start_run.return_value = fake_run
    fake_mlflow.get_tracking_uri.return_value = "sqlite:///mlflow.db"

    fake_infer_signature = MagicMock(return_value="signature")

    with patch.object(train.models, "Sequential", return_value=fake_model) as mock_sequential, \
         patch.object(train, "mlflow", fake_mlflow), \
         patch.object(train, "infer_signature", fake_infer_signature):

        result = train.train_model(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            str(tmp_path / "cats_dogs_cnn.keras"),
        )

    # train_model does not explicitly return the model.
    assert result is None

    mock_sequential.assert_called_once()

    args = mock_sequential.call_args.args[0]

    # 4 convolutional layers, 4 pooling layers, Flatten, Dense, Dropout, Dense.
    assert len(args) == 12

    assert args[0].filters == 32
    assert args[1].pool_size == (2, 2)
    assert args[2].filters == 64
    assert args[4].filters == 128
    assert args[6].filters == 256
    assert args[-1].units == 1
    assert args[-1].activation.__name__ == "sigmoid"

    fake_model.compile.assert_called_once_with(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    fake_model.fit.assert_called_once()
    fake_model.evaluate.assert_called_once()
    fake_model.save.assert_called_once()

    fake_mlflow.log_params.assert_called_once()
    fake_mlflow.tensorflow.log_model.assert_called_once()


def test_train_model_uses_five_epochs():
    fake_model = MagicMock()

    class DummyHistory:
        history = {
            "loss": [0.8] * 5,
            "accuracy": [0.7] * 5,
            "val_loss": [0.75] * 5,
            "val_accuracy": [0.72] * 5,
        }

    fake_model.fit.return_value = DummyHistory()
    fake_model.evaluate.return_value = (0.5, 0.8)
    fake_model.predict.return_value = np.array([[0.8]], dtype=np.float32)

    fake_run = MagicMock()
    fake_run.__enter__.return_value = fake_run
    fake_run.info.run_id = "test"

    fake_mlflow = MagicMock()
    fake_mlflow.start_run.return_value = fake_run

    with patch.object(train.models, "Sequential", return_value=fake_model), \
         patch.object(train, "mlflow", fake_mlflow), \
         patch.object(train, "infer_signature", return_value="signature"):

        train.train_model(
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    fake_model.fit.assert_called_once()
    assert fake_model.fit.call_args.kwargs["epochs"] == 5

    # One final test metric plus one metric record per training epoch.
    assert fake_mlflow.log_metrics.call_count >= 6
