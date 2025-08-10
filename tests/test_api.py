import os

os.environ["TEST_MODE"] = "1"  # bypass MLflow in CI

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_ok():
    payload = {
        "MedInc": 8.3,
        "HouseAge": 41.0,
        "AveRooms": 6.98,
        "AveBedrms": 1.02,
        "Population": 322.0,
        "AveOccup": 2.55,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "predicted_median_house_value" in r.json()
