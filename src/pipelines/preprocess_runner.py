import sys
from pathlib import Path

from src.data.preprocess_03 import load_raw_data, preprocess, split_and_save

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


def main():
    """Runs the preprocessing pipeline."""
    raw_data_path = "data/raw/california_housing.csv"
    output_dir = "data/processed"

    print(f"Loading raw data from: {raw_data_path}")
    df = load_raw_data(raw_data_path)

    print("Preprocessing data...")
    df_clean = preprocess(df)

    print(f"Saving processed data to: {output_dir}")
    split_and_save(df_clean, output_dir)
    print("✅ Preprocessing complete.")


if __name__ == "__main__":
    main()
