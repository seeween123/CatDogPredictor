
# Pytest tests for the Cats vs Dogs CNN pipeline

These tests cover:

- `preprocessing(2).py`
  - image copying
  - class directory assignment
  - JPG filtering
  - 224x224 image generators
  - batch size and binary classification
  - test generator shuffle setting
- `train(3).py`
  - CNN layer structure
  - optimizer/loss/metrics
  - five training epochs
  - evaluation and MLflow logging
- `predict(4).py`
  - Dog prediction at probability >= 0.5
  - Cat prediction at probability < 0.5
  - image resizing and normalization
- `make_dataset(2).py`
  - current source-code validity issue
  - expected Kaggle dataset and ZIP names

## Run

From the directory containing the four source files:

```bash
pip install pytest
pytest -v tests
```

The preprocessing and training tests require the project's TensorFlow and
MLflow dependencies.

## Important source issues found

Two uploaded files currently need small fixes before they can be used as
normal Python modules:

1. `make_dataset(2).py` contains Jupyter/IPython `!` commands:
   `!shutil.copy(...)` and `!kaggle datasets download ...`.
   These are not valid Python syntax in a `.py` file.

2. `predict(4).py` defines `predict_model(...)` but calls
   `predict_cat_or_dog(...)` at module level. It also executes a prediction
   immediately when imported. The prediction tests therefore extract the
   function without executing the example call.

Recommended production fix for `predict(4).py`:

```python
if __name__ == "__main__":
    image_path = r"../dataset/Processed/cats_dogs_split/test/cat/42.jpg"
    predict_model(image_path)
```
