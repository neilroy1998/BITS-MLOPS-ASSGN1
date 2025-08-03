# api/main.py
import os
from urllib.request import Request

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from prometheus_client import Histogram  # New import for custom metric
from prometheus_fastapi_instrumentator import Instrumentator  # New import
from pydantic import BaseModel, ConfigDict

# --- 1. SETUP, METRICS & MODEL LOADING ---

# Provides metadata for the auto-generated API docs
app = FastAPI(
    title="California House Price Prediction API",
    description="API for predicting house prices using the California Housing dataset.",
    version="1.0.0",
)


# --- GLOBAL EXCEPTION HANDLER ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches all unhandled exceptions and returns a standard JSON error."""
    # For production, you would log the full exception details here
    print(f"An unhandled exception occurred: {repr(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred on the server.",
        },
    )


# Add Prometheus metrics
# This exposes a /metrics endpoint automatically
Instrumentator().instrument(app).expose(app)

# Create a custom histogram to track the distribution of predicted values
PREDICTION_HISTOGRAM = Histogram(
    "predicted_house_value", "Distribution of predicted house values ($100k)"
)

# Load environment variables from .env file
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_ALIAS = "Production"

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"🔄 Loading model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'...")

try:
    client = MlflowClient()
    latest_version_info = client.get_model_version_by_alias(
        name=MODEL_NAME, alias=MODEL_ALIAS
    )
    model_version = latest_version_info.version
    model_uri = f"models:/{MODEL_NAME}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"✅ Model version {model_version} loaded successfully from {model_uri}!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model: {e}") from e


# --- 2. INPUT SCHEMA using Pydantic ---


class HouseFeatures(BaseModel):
    """Defines the input features for a single prediction."""

    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

    # Pydantic v2 syntax for providing an example in the docs
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.9841,
                "AveBedrms": 1.0238,
                "Population": 322.0,
                "AveOccup": 2.5555,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }
    )


# --- 3. API ENDPOINTS ---


@app.get("/")
def read_root():
    """Returns a welcome message for the API root."""
    return {"message": "Welcome to the House Price Prediction API!"}


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HouseFeatures):
    """Takes house features and returns a predicted house value."""
    input_df = pd.DataFrame([features.model_dump()])

    prediction = model.predict(input_df)

    predicted_value = prediction[0]

    # Observe the predicted value with our custom Prometheus metric
    PREDICTION_HISTOGRAM.observe(predicted_value)

    return {"predicted_median_house_value": predicted_value}
