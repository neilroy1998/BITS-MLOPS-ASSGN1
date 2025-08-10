# California Housing MLOps — System Architecture

This project implements an end-to-end MLOps workflow: data versioning with **DVC**, experiment tracking & model registry with **MLflow**, a **FastAPI** prediction service with **Pydantic** validation, and **Prometheus + Grafana** monitoring. Retraining is triggered by drift checks (bonus).

---

## Components

- **Data & Pipeline**:  
  - `data/raw/` → `src/data/preprocess_03.py` → `data/processed/` (tracked by **DVC**, see `dvc.yaml` stages: `preprocess`, `train`, `register`)
- **Experiment Tracking & Model Registry**:  
  - **MLflow server** (`start_mlflow_ui.sh`), SQLite backend `mlflow.db`, artifacts in `./mlartifacts/`  
  - Best model registered as **`HousePriceModel`** with alias **`Production`**
- **Serving (FastAPI)**:  
  - `api/main.py`: `/predict`, `/health`, `/metrics`  
  - Input validation with **Pydantic** (`HouseFeatures`)  
  - Structured logs via **Loguru** → `logs/app.log`  
  - Optional request logging to SQLite → `monitoring/predictions.db`
- **Monitoring**:  
  - **Prometheus** scrapes `/metrics` (FastAPI + custom histogram)  
  - **Grafana** dashboard provisioned from `monitoring/grafana/`
- **Docker**:  
  - AIO image (`docker/Dockerfile`) runs API + MLflow + Prometheus + Grafana via **supervisord**  
  - `docker-compose.yml` available for multi-service workflows
- **CI**:  
  - GitHub Actions builds a CI image and runs tests with `TEST_MODE=1` (dummy model)

> Key env: `MLFLOW_TRACKING_URI=http://127.0.0.1:5002`, `MODEL_NAME=HousePriceModel`

---

## High-Level System Diagram

```mermaid
flowchart LR
  user[Client] -->|/predict JSON| api[FastAPI Service]
  api -->|Pydantic validate| api
  api -->|load model by alias 'Production'| mlflow_api[MLflow Tracking/Registry]
  api -->|/metrics| metrics[/Prometheus metrics endpoint/]
  api -->|log prediction| predDB[(SQLite predictions.db)]
  prom[Prometheus] -->|datasource| graf[Grafana]
  prom <-->|scrape /metrics| metrics
  subgraph Tracking
    mlflow_api <-->|runs, artifacts| train[Train code (MLflow autolog)]
    store[(SQLite backend: mlflow.db)]
    arts[(mlartifacts/)]
    mlflow_api --- store
    mlflow_api --- arts
  end
````

---

## DVC Pipeline (Training & Registration)

```mermaid
flowchart LR
  raw[data/raw/california_housing.csv] --> pre[DVC: preprocess]
  pre --> proc[data/processed/{train,test}.parquet]
  proc --> train[DVC: train_multiple_models.py (LR/DT/RF/GB)]
  train --> reg[DVC: register_best_model.py]
  reg -->|alias=Production| registry[(MLflow Model Registry)]
```

**Notes**

* **Model selection**: top run by `custom_rmse` (tie-break `custom_r2_score`) → registered → alias `Production`.
* **Serving contract**: API loads the current `Production` version at startup via MLflow client.
* **Observability**: `prometheus_fastapi_instrumentator` + custom `predicted_house_value` histogram; Grafana uses provisioned Prometheus datasource.

---

## Deployment Modes

* **All-in-One (local demo)**: single container with API + MLflow + Prometheus + Grafana via `supervisord` (see `docker/Dockerfile` and `monitoring/`).
* **Compose / Split Services**: use `docker-compose.yml` to run services separately if needed.

---

## Security & Reproducibility (at a glance)

* Deterministic pipeline via **DVC** + `dvc.lock`
* Environment captured via **uv** + `pyproject.toml`/`uv.lock`
* No secrets in repo; use `.env` (see `.env.example`); logs rotated/compressed.

---
