from src.data.preprocess_03 import load_raw_data, preprocess, split_and_save


def main():
    raw_data_path = "../../data/raw/california_housing.csv"
    output_dir = "../../data/processed"

    df = load_raw_data(raw_data_path)
    df_clean = preprocess(df)
    split_and_save(df_clean, output_dir)


if __name__ == "__main__":
    main()
