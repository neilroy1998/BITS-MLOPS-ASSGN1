# api/main.py
import os
import sqlite3  # New import
import sys
from datetime import datetime  # New import
from pathlib import Path  # New import
from urllib.request import Request

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from mlflow.tracking import MlflowClient
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict

# --- 1. LOGGER & DB CONFIGURATION ---
logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    serialize=True,
    level="INFO",
)

# Path to the SQLite database
DB_PATH = Path(__file__).parent.parent / "monitoring" / "predictions.db"

# --- 2. API & MODEL SETUP ---
app = FastAPI(
    title="California House Price Prediction API",
    description="API for predicting house prices using the California Housing dataset.",
    version="1.0.0",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception for request {request.method} {request.url}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred on the server.",
        },
    )


Instrumentator().instrument(app).expose(app)
PREDICTION_HISTOGRAM = Histogram(
    "predicted_house_value", "Distribution of predicted house values ($100k)"
)

load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_ALIAS = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"Attempting to load model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'...")
try:
    client = MlflowClient()
    latest_version_info = client.get_model_version_by_alias(
        name=MODEL_NAME, alias=MODEL_ALIAS
    )
    model_version = latest_version_info.version
    model_uri = f"models:/{MODEL_NAME}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info(
        f"✅ Model version {model_version} loaded successfully from {model_uri}."
    )
except Exception as e:
    logger.exception("❌ Failed to load model from MLflow.")
    raise RuntimeError(f"Could not load model: {e}") from e


# --- 3. DATABASE LOGGING FUNCTION ---
def log_prediction_to_db(features: dict, predicted_value: float):
    """Logs the input features and prediction result to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        insert_query = """
                       INSERT INTO predictions (timestamp, MedInc, HouseAge, AveRooms, AveBedrms, Population, \
                                                AveOccup, Latitude, Longitude, predicted_value) \
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?); \
                       """

        # Prepare data tuple in the correct order
        data_tuple = (
            datetime.utcnow(),
            features["MedInc"],
            features["HouseAge"],
            features["AveRooms"],
            features["AveBedrms"],
            features["Population"],
            features["AveOccup"],
            features["Latitude"],
            features["Longitude"],
            predicted_value,
        )

        cursor.execute(insert_query, data_tuple)
        conn.commit()
    except sqlite3.Error as e:
        # Log the error but don't crash the API
        logger.error(f"Failed to log prediction to database. Error: {e}")
    finally:
        if conn:
            conn.close()


# --- 4. INPUT SCHEMA ---
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
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


# --- 5. API ENDPOINTS ---
@app.get("/")
def read_root():
    logger.info("Root endpoint was accessed.")
    return {"message": "Welcome to the House Price Prediction API!"}


@app.get("/health")
def health_check():
    logger.info("Health check performed.")
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HouseFeatures):
    feature_dict = features.model_dump()
    logger.info(f"Received prediction request with features: {feature_dict}")

    input_df = pd.DataFrame([feature_dict])
    prediction = model.predict(input_df)
    predicted_value = prediction[0]

    PREDICTION_HISTOGRAM.observe(predicted_value)
    logger.info(f"Prediction successful. Result: ${predicted_value:,.2f}")

    # Log the result to our database
    log_prediction_to_db(feature_dict, predicted_value)

    return {"predicted_median_house_value": predicted_value}
