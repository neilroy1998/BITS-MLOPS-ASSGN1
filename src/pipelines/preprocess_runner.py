import sys
from pathlib import Path

from src.data.preprocess_03 import load_raw_data, preprocess, split_and_save

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


def main():
    raw_data_path = "data/raw/california_housing.csv"
    output_dir = "data/processed"

    df = load_raw_data(raw_data_path)
    df_clean = preprocess(df)
    split_and_save(df_clean, output_dir)


if __name__ == "__main__":
    main()
