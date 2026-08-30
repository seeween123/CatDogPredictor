import os
import shutil
import subprocess
import zipfile

def load_data(output_path: str = "dataset/Raw"):
    """Download and extract the Cat/Dog dataset from Kaggle."""
    kaggle_dir = os.path.join(os.environ["USERPROFILE"], ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    source_kaggle_json = os.path.join("..", "kaggle.json")
    destination_kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(source_kaggle_json):
        raise FileNotFoundError(
            f"Kaggle credentials not found: {source_kaggle_json}"
        )

    shutil.copy(source_kaggle_json, destination_kaggle_json)
    os.makedirs(output_path, exist_ok=True)

    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "bhavikjikadara/dog-and-cat-classification-dataset",
        ],
        check=True,
    )

    zip_path = "dog-and-cat-classification-dataset.zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_path)


if __name__ == "__main__":
    load_data()
