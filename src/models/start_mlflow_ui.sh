#!/bin/bash
# scripts/start_mlflow.sh

# Load env vars from .env
set -a
source .env
set +a

echo "🌐 Starting MLflow on $MLFLOW_TRACKING_URI"

mlflow ui \
  --backend-store-uri ./mlruns \
  --default-artifact-root ./mlartifacts \
  --port "$MLFLOW_PORT"
