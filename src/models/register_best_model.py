# src/models/register_and_promote.py ...

import os
import warnings
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Suppress future warnings from MLflow for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. SETUP ---
# Load environment variables from .env file
load_dotenv()

# Configure MLflow to connect to your tracking server
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()
model_name = os.environ["MODEL_NAME"]

print(f"🔄 Starting process for model: {model_name}")

# --- 2. FIND THE BEST RUN ---
# Get the 'default' experiment and search for the top-performing run
experiment = client.get_experiment_by_name("default")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.custom_rmse ASC", "metrics.custom_r2_score DESC"],
    max_results=1,
)

if not runs:
    raise Exception("❌ No runs found to register.")

best_run = runs[0]
print(f"🏆 Found best run: {best_run.data.tags.get('mlflow.runName')}")
print(
    f"📉 RMSE: {best_run.data.metrics['custom_rmse']:.4f}, 📈 R²: {best_run.data.metrics['custom_r2_score']:.4f}"
)

# --- 3. REGISTER THE MODEL ---
# Create the model URI from the best run's ID
model_uri = f"runs:/{best_run.info.run_id}/model"

# Register the model and wait for the registration to complete
registered_version = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
    await_registration_for=300,  # Wait up to 5 minutes
)
print(f"✅ Registered model '{model_name}' with version {registered_version.version}")

# --- 4. PROMOTE THE NEWLY REGISTERED MODEL ---
# Set the "Production" alias for the version we just registered
client.set_registered_model_alias(
    name=model_name, alias="Production", version=registered_version.version
)
print(f"🚀 Promoted version {registered_version.version} to 'Production' alias.")
print("✨ Process complete.")
