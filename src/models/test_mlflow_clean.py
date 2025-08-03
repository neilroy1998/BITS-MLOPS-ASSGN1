import os

import mlflow
from dotenv import load_dotenv

load_dotenv()  # 👈 Load MLFLOW_TRACKING_URI from .env

print(f"📡 Tracking URI from env: {mlflow.get_tracking_uri()}")

mlflow.set_experiment("default")  # 👈 This line fixes the RESOURCE_DOES_NOT_EXIST issue

with mlflow.start_run(run_name="default-run"):
    mlflow.log_param("version", "back-to-default")
    mlflow.log_metric("score", 0.99)
