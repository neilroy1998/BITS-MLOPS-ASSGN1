import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load MLFLOW_TRACKING_URI from .env if not already set
load_dotenv()

print(f"📡 MLflow Tracking URI: {mlflow.get_tracking_uri()}")
mlflow.set_experiment("default")  # Create/use 'default' experiment
mlflow.sklearn.autolog()

# Resolve paths to processed data
ROOT = Path(__file__).resolve().parents[2]
train_path = ROOT / "data" / "processed" / "train.parquet"
test_path = ROOT / "data" / "processed" / "test.parquet"

print(f"📁 Loading train data from: {train_path}")
print(f"📁 Loading test data from: {test_path}")

# Load train/test data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

X_train = train_df.drop("MedHouseVal", axis=1)
y_train = train_df["MedHouseVal"]
X_test = test_df.drop("MedHouseVal", axis=1)
y_test = test_df["MedHouseVal"]

# Train & log
with mlflow.start_run(run_name="LinearRegression-Baseline"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ R²: {r2:.4f}")
