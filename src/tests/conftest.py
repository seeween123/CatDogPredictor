
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(filename: str, module_name: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
