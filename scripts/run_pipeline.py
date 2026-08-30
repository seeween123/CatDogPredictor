# test_pipeline_phase1.py
import os

# Make sure Python can find your src package
import sys

sys.path.append(os.path.abspath("src"))

from data.make_dataset import load_data
from data.preprocessing import preprocess_data
from models.train import train_model


def main():
    print("=== Testing Phase 1: Load → Preprocess → Build Features ===")

    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_data()
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(3))

    # 2. Preprocess
    print("\n[2] Preprocessing data...")
    train_generator, val_generator, test_generator = preprocess_data()

    print(f"train_generator after preprocessing. Shape: {train_generator.shape}")
    print(train_generator.head(3))
    print(f"Data after preprocessing. Shape: {train_generator.shape}")
    print(val_generator.head(3))
    print(f"Data after preprocessing. Shape: {val_generator.shape}")
    print(test_generator.head(3))
    print(f"Data after preprocessing. Shape: {test_generator.shape}")

    train_model(train_generator, val_generator, test_generator)

    print("\n✅ ML Pipeline completed successfully!")


if __name__ == "__main__":
    main()
