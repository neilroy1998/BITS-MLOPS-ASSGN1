# California Housing — End-to-End MLOps

Minimal, reproducible MLOps stack:

* **Data & Pipeline:** DVC (`dvc.yaml`) → preprocess → train → register
* **Tracking/Registry:** MLflow server (SQLite backend), model alias **Production**
* **Serving:** FastAPI (`/predict`, `/health`, `/metrics`) with Pydantic validation
* **Monitoring:** Prometheus scrape + Grafana dashboard (provisioned)
* **CI:** GitHub Actions builds CI image and runs tests with `TEST_MODE=1`
* **Docker:** All-in-one image (API+MLflow+Prometheus+Grafana) or split via Compose

---

## Quickstart (Mac, uv)

```bash
# 1) env
cp .env .env.local 2>/dev/null || true   # optional copy
# Ensure .env has:
# MLFLOW_TRACKING_URI=http://127.0.0.1:5002
# MLFLOW_PORT=5002
# MODEL_NAME=HousePriceModel

# 2) install project
uv pip install -e .

# 3) start MLflow locally (SQLite backend + local artifacts)
bash src/models/start_mlflow_ui.sh

# 4) reproduce the full pipeline (preprocess -> train -> register)
dvc repro

# 5) run API (loads model by alias 'Production')
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

* `GET /health` → `{status: "ok"}`
* `POST /predict` → `{"predicted_median_house_value": <float>}`
* `GET /metrics` → Prometheus format
* OpenAPI spec is at `docs/api_spec.json` (exported offline).

Example:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"MedInc":8.3,"HouseAge":41,"AveRooms":6.98,"AveBedrms":1.02,"Population":322,"AveOccup":2.55,"Latitude":37.88,"Longitude":-122.23}'
```

---

## Project Layout

```
data/{raw,processed}         notebooks/                 src/{data,models,pipelines,retraining,utils}
api/main.py                  monitoring/{prometheus.yml,grafana/...}
docs/{eda/,architecture.md,api_spec.json}  logs/       tests/
docker/{Dockerfile,supervisor/...}         dvc.yaml    dvc.lock
```

Key bits:

* `src/data/preprocess_03.py` → cleans/features → `data/processed/{train,test}.parquet`
* `src/models/train_multiple_models.py` → logs runs to MLflow
* `src/models/register_best_model.py` → picks best by `custom_rmse` (tie: `custom_r2_score`) and sets alias **Production**
* `api/main.py` → Pydantic schema + Prometheus instrumentation + optional SQLite logging to `monitoring/predictions.db`
* `monitoring/grafana/...` → provisioned datasource + dashboard JSON

---

## Reproducibility

```bash
# Data versioning
dvc status
dvc repro

# OpenAPI (offline export)
uv run python - <<'PY'
import os, sys, json
from pathlib import Path
os.environ["TEST_MODE"]="1"; sys.path.append(str(Path(".").resolve()))
from api.main import app
Path("docs").mkdir(parents=True, exist_ok=True)
Path("docs/api_spec.json").write_text(json.dumps(app.openapi(), indent=2))
print("Wrote docs/api_spec.json")
PY
```

---

## Monitoring (local demo)

All-in-one container (API+MLflow+Prometheus+Grafana via supervisord):

```bash
# Build (multi-arch ready; set your Docker Hub user)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile \
  --build-arg INSTALL_PROFILE=full \
  -t bitswilp2023ac05605/housing-allinone:full \
  .

# Run
docker run --rm --name housing-aio \
  -p 8000:8000 -p 5002:5002 -p 9090:9090 -p 3000:3000 \
  --env-file .env \
  <docker-username>/housing-allinone:full
```

* API: [http://localhost:8000](http://localhost:8000)
* MLflow: [http://localhost:5002](http://localhost:5002)
* Prometheus: [http://localhost:9090](http://localhost:9090)
* Grafana: [http://localhost:3000](http://localhost:3000) (admin/admin; provisioned dashboard loaded)

> Compose set-up is also available via `docker-compose.yml` if you prefer split services.

---

## Testing

```bash
# Local tests (uses TEST_MODE=1 → dummy model)
TEST_MODE=1 uv run pytest -q
```

CI runs in GitHub Actions: builds a CI image (`INSTALL_PROFILE=ci`) and runs the same tests.

---

## Environment

| Variable              | Default                 | Purpose                                  |
| --------------------- | ----------------------- | ---------------------------------------- |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5002` | MLflow server base URL                   |
| `MLFLOW_PORT`         | `5002`                  | MLflow server port                       |
| `MODEL_NAME`          | `HousePriceModel`       | Model registry name                      |
| `TEST_MODE`           | unset                   | If `1`, API uses a dummy model for tests |

---

## Docker Hub & Links (fill when ready)

* GitHub repo: **TBD**
* Docker image: **TBD** (e.g., `docker pull bitswilp2023ac05605/housing-allinone:full`)
* Demo video: **TBD**
* One-pager architecture: `docs/architecture.md`

---
