# api/main.py
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict

load_dotenv()
TEST_MODE = os.getenv("TEST_MODE") == "1" or bool(os.getenv("PYTEST_CURRENT_TEST"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_ALIAS = "Production"

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)
if not TEST_MODE:
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        serialize=True,
        level="INFO",
    )

DB_PATH = (
    None
    if TEST_MODE
    else Path(__file__).parent.parent / "monitoring" / "predictions.db"
)

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

# Model: dummy in tests, MLflow in normal runs
if TEST_MODE:

    class _DummyModel:
        def predict(self, df: pd.DataFrame):
            return [float(df["MedInc"].iat[0])]

    model = _DummyModel()
    logger.info("Loaded dummy model (TEST_MODE=1).")
else:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(
        f"Attempting to load model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'..."
    )
    try:
        client = MlflowClient()
        latest = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
        model_uri = f"models:/{MODEL_NAME}/{latest.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded model version {latest.version} from {model_uri}.")
    except Exception as e:
        logger.exception("Failed to load model from MLflow.")
        raise RuntimeError(f"Could not load model: {e}") from e


def log_prediction_to_db(features: dict, predicted_value: float):
    """Log input and prediction to SQLite if enabled."""
    if DB_PATH is None:
        return
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions
            (timestamp, MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, predicted_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                features["MedInc"],
                features["HouseAge"],
                features["AveRooms"],
                features["AveBedrms"],
                features["Population"],
                features["AveOccup"],
                features["Latitude"],
                features["Longitude"],
                predicted_value,
            ),
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to log prediction to database. Error: {e}")
    finally:
        if conn:
            conn.close()


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


@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HouseFeatures):
    feature_dict = features.model_dump()
    input_df = pd.DataFrame([feature_dict])
    predicted_value = float(model.predict(input_df)[0])
    PREDICTION_HISTOGRAM.observe(predicted_value)
    log_prediction_to_db(feature_dict, predicted_value)
    return {"predicted_median_house_value": predicted_value}
