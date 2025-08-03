from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# Load .env if needed
load_dotenv()
mlflow.set_experiment("default")
mlflow.sklearn.autolog()

# Paths
ROOT = Path(__file__).resolve().parents[2]
train_path = ROOT / "data" / "processed" / "train.parquet"
test_path = ROOT / "data" / "processed" / "test.parquet"

print(f"📁 Loading train data from: {train_path}")
print(f"📁 Loading test data from: {test_path}")

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)
X_train = train_df.drop("MedHouseVal", axis=1)
y_train = train_df["MedHouseVal"]
X_test = test_df.drop("MedHouseVal", axis=1)
y_test = test_df["MedHouseVal"]

# Models to train
models = {
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}

# Train & log each model
for name, model in models.items():
    with mlflow.start_run(run_name=f"{name}-Model"):
        print(f"🔁 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        print(f"✅ {name} RMSE: {rmse:.4f}")
        print(f"✅ {name} R²: {r2:.4f}")

        # Log extra metrics explicitly
        mlflow.log_metric("custom_rmse", rmse)
        mlflow.log_metric("custom_r2_score", r2)
