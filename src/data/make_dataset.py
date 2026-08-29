import os
import shutil

def load_data(output_path: str = "dataset/Raw"):
    """
    Download the Cat, Dog dataset from Kaggle.

    Returns
    -------
    None
    """

    kaggle_dir = os.path.join(os.environ['USERPROFILE'], '.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    print(f"Kaggle directory created at: {kaggle_dir}")

    !shutil.copy('../kaggle.json', os.path.join(kaggle_dir, 'kaggle.json'))
    !kaggle datasets download -d bhavikjikadara/dog-and-cat-classification-dataset

    import zipfile

    with zipfile.ZipFile("dog-and-cat-classification-dataset.zip", "r") as zip_ref:
        zip_ref.extractall(output_path)

    return


if __name__ == "__main__":
    load_data()
