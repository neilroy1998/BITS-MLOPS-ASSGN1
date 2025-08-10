#!/usr/bin/env bash
set -euo pipefail

# Load env
set -a
source .env
set +a

# One URI for everything (works locally now, and inside a single-container later)
export MLFLOW_TRACKING_URI="http://127.0.0.1:${MLFLOW_PORT:-5002}"

mkdir -p mlartifacts

echo "🌐 Starting MLflow tracking server at $MLFLOW_TRACKING_URI"

# Use a SQLite backend so Model Registry & aliases work reliably
uv run mlflow server \
  --host 127.0.0.1 \
  --port "${MLFLOW_PORT:-5002}" \
  --backend-store-uri "sqlite:///mlflow.db" \
  --serve-artifacts \
  --artifacts-destination "./mlartifacts"
