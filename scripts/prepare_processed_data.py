import pandas as pd
from pathlib import Path

# make src importable
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import preprocess_data
from src.features.build_features import build_features

RAW = Path("data/raw/heart_disease.csv")
OUT = Path("data/processed/heart_disease_processed.csv")


def main():
    # load raw
    df = pd.read_csv(RAW)

    # preprocess
    df = preprocess_data(df, target_col="target")

    # sanity checks
    assert df["target"].isna().sum() == 0, "target has NaNs after preprocess"
    assert set(df["target"].unique()) <= {0, 1}, "target not 0/1 after preprocess"

    # 4) features
    df_processed = build_features(df, target_col="target")

    # 5) save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(OUT, index=False)
    print(f"✅ Processed dataset saved to {OUT} | Shape: {df_processed.shape}")


if __name__ == "__main__":
    main()
