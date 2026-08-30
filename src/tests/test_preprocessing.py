
from pathlib import Path
from unittest.mock import patch

from conftest import load_module

preprocessing = load_module("preprocessing(2).py", "preprocessing")


def test_copy_images_copies_cat_and_dog_to_correct_directories(tmp_path):
    source_cat = tmp_path / "cat1.jpg"
    source_dog = tmp_path / "dog1.jpg"
    source_cat.write_bytes(b"cat")
    source_dog.write_bytes(b"dog")

    base_dir = tmp_path / "processed"

    preprocessing.copy_images(
        [str(source_cat), str(source_dog)],
        [0, 1],
        "train",
        str(base_dir),
    )

    assert (base_dir / "train" / "cat" / "cat1.jpg").read_bytes() == b"cat"
    assert (base_dir / "train" / "dog" / "dog1.jpg").read_bytes() == b"dog"


def test_copy_images_ignores_copy_errors(tmp_path):
    with patch.object(
        preprocessing.shutil,
        "copy",
        side_effect=OSError("copy failed"),
    ):
        # The function intentionally suppresses copy exceptions.
        preprocessing.copy_images(
            ["missing.jpg"],
            [0],
            "train",
            str(tmp_path / "processed"),
        )


def test_preprocess_data_uses_expected_generators_and_split_directories(tmp_path):
    source = tmp_path / "raw"
    cat_dir = source / "Cat"
    dog_dir = source / "Dog"
    cat_dir.mkdir(parents=True)
    dog_dir.mkdir(parents=True)

    for i in range(4):
        (cat_dir / f"cat_{i}.jpg").write_bytes(b"cat")
        (dog_dir / f"dog_{i}.jpg").write_bytes(b"dog")

    output = tmp_path / "processed"

    # The uploaded implementation calls copy_images without forwarding
    # `base_dir`. Patch it here so this test can isolate preprocess_data's
    # splitting and generator configuration.
    def copy_to_requested_output(imgs, labels, split, base_dir="../dataset/Processed/cats_dogs_split"):
        for img, label in zip(imgs, labels):
            cls = "cat" if label == 0 else "dog"
            destination = Path(output) / split / cls / Path(img).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(img).read_bytes())

    with patch.object(preprocessing, "copy_images", side_effect=copy_to_requested_output):
        train_gen, val_gen, test_gen = preprocessing.preprocess_data(
            str(source),
            str(output),
        )

    assert train_gen.target_size == (224, 224)
    assert val_gen.target_size == (224, 224)
    assert test_gen.target_size == (224, 224)

    assert train_gen.batch_size == 32
    assert val_gen.batch_size == 32
    assert test_gen.batch_size == 32

    assert train_gen.class_mode == "binary"
    assert val_gen.class_mode == "binary"
    assert test_gen.class_mode == "binary"
    assert test_gen.shuffle is False

    assert train_gen.samples == 6
    assert val_gen.samples == 1
    assert test_gen.samples == 1

    for split in ("train", "val", "test"):
        for cls in ("cat", "dog"):
            assert (output / split / cls).is_dir()


def test_preprocess_data_only_reads_jpg_files(tmp_path):
    source = tmp_path / "raw"
    (source / "Cat").mkdir(parents=True)
    (source / "Dog").mkdir(parents=True)

    (source / "Cat" / "cat.jpg").write_bytes(b"x")
    (source / "Cat" / "cat.png").write_bytes(b"x")
    (source / "Dog" / "dog.jpg").write_bytes(b"x")
    (source / "Dog" / "dog.txt").write_bytes(b"x")

    output = tmp_path / "processed"

    train_gen, val_gen, test_gen = preprocessing.preprocess_data(
        str(source),
        str(output),
    )

    total = (
        train_gen.samples
        + val_gen.samples
        + test_gen.samples
    )

    assert total == 2
