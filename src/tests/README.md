
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

