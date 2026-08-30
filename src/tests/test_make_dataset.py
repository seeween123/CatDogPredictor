
import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "make_dataset(2).py"


def test_make_dataset_source_is_not_valid_python_until_notebook_magics_are_removed():
    """
    Documents a current source-code issue.

    The uploaded file contains IPython `!` commands. pytest cannot import it
    as a normal Python module until those commands are replaced with Python
    APIs/subprocess calls.
    """
    with pytest.raises(SyntaxError):
        ast.parse(SOURCE.read_text())


def test_expected_kaggle_dataset_identifier_is_present():
    source = SOURCE.read_text()
    assert "bhavikjikadara/dog-and-cat-classification-dataset" in source
    assert "dog-and-cat-classification-dataset.zip" in source


def test_load_data_contract_is_documented_in_source():
    source = SOURCE.read_text()
    assert "def load_data(output_path: str = \"dataset/Raw\")" in source
    assert "zip_ref.extractall(output_path)" in source
