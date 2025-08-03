# src/models/register_best_model.py

import os
import warnings
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

experiment = client.get_experiment_by_name("default")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.custom_rmse ASC", "metrics.custom_r2_score DESC"],
    max_results=1,
)

if not runs:
    raise Exception("❌ No runs found to register.")

best_run = runs[0]
print(f"🏆 Best run: {best_run.data.tags.get('mlflow.runName')}")
print(
    f"📉 RMSE: {best_run.data.metrics['custom_rmse']}, 📈 R²: {best_run.data.metrics['custom_r2_score']}"
)

model_uri = f"runs:/{best_run.info.run_id}/model"
model_name = os.environ["MODEL_NAME"]

# Register the model
result = mlflow.register_model(
    model_uri=model_uri, name=model_name, await_registration_for=30
)
print(f"✅ Registered {model_name}, version {result.version}")
