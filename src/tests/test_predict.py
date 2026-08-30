
import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "predict(4).py"


def load_predict_function():
    """
    Extract only predict_model from the uploaded file.

    The current source file contains an unconditional example call at
    module level, so importing the complete module would execute that
    example. AST extraction lets these tests exercise the function itself.
    """
    tree = ast.parse(SOURCE.read_text())

    function_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "predict_model"
    )

    module = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="matplotlib.pyplot", asname="plt")]),
            ast.Import(names=[ast.alias(name="numpy", asname="np")]),
            ast.ImportFrom(
                module="tensorflow.keras.models",
                names=[ast.alias(name="load_model", asname=None)],
                level=0,
            ),
            ast.ImportFrom(
                module="tensorflow.keras.utils",
                names=[
                    ast.alias(name="img_to_array", asname=None),
                    ast.alias(name="load_img", asname=None),
                ],
                level=0,
            ),
            function_node,
        ],
        type_ignores=[],
    )

    namespace = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["predict_model"], namespace


def test_predict_model_returns_dog_for_probability_at_or_above_half():
    predict_model, namespace = load_predict_function()

    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.8]], dtype=np.float32)

    fake_image = MagicMock()
    fake_array = np.ones((224, 224, 3), dtype=np.float32)

    with patch.dict(
        namespace,
        {
            "load_model": MagicMock(return_value=fake_model),
            "load_img": MagicMock(return_value=fake_image),
            "img_to_array": MagicMock(return_value=fake_array),
        },
    ):
        label, confidence = predict_model("dog.jpg")

    assert label == "Dog"
    assert confidence == 0.8
    fake_model.predict.assert_called_once()


def test_predict_model_returns_cat_for_probability_below_half():
    predict_model, namespace = load_predict_function()

    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.2]], dtype=np.float32)

    with patch.dict(
        namespace,
        {
            "load_model": MagicMock(return_value=fake_model),
            "load_img": MagicMock(return_value=MagicMock()),
            "img_to_array": MagicMock(
                return_value=np.ones((224, 224, 3), dtype=np.float32)
            ),
        },
    ):
        label, confidence = predict_model("cat.jpg")

    assert label == "Cat"
    assert confidence == 0.8


def test_predict_model_preprocesses_image_to_224_rgb_scaled_input():
    predict_model, namespace = load_predict_function()

    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.5]], dtype=np.float32)

    raw = np.full((224, 224, 3), 255, dtype=np.float32)
    img_to_array = MagicMock(return_value=raw)

    with patch.dict(
        namespace,
        {
            "load_model": MagicMock(return_value=fake_model),
            "load_img": MagicMock(return_value=MagicMock()),
            "img_to_array": img_to_array,
        },
    ):
        predict_model("image.jpg")

    namespace["load_img"].assert_called_once_with(
        "image.jpg",
        target_size=(224, 224),
    )

    # Verify that the model receives a batch dimension and values scaled to [0, 1].
    model_input = fake_model.predict.call_args.args[0]

    assert model_input.shape == (1, 224, 224, 3)
    assert np.allclose(model_input, 1.0)
