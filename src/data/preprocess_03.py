import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(inplace=True)

    # I got these from the EDA step (describe)
    df = df[df["MedInc"] < 20]
    df = df[df["AveRooms"] < 20]
    df = df[df["AveOccup"] < 10]

    df["Population"] = df["Population"].apply(lambda x: np.log1p(x))

    # Separate target column
    target_col = "MedHouseVal"
    features = df.drop(columns=[target_col])
    target = df[target_col]

    # Standard scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

    # Combine back with target
    df_processed = pd.concat(
        [features_scaled_df, target.reset_index(drop=True)], axis=1
    )
    return df_processed


def split_and_save(df: pd.DataFrame, output_dir: str, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    print(f"✅ Data saved to: {output_dir}")


if __name__ == "__main__":
    raw_data_path = "../../data/raw/california_housing.csv"
    output_dir = "../../data/processed"

    df = load_raw_data(raw_data_path)
    df_clean = preprocess(df)
    split_and_save(df_clean, output_dir)
