# Dockerfile — one container: MLflow + FastAPI + Prometheus + Grafana
FROM python:3.12-slim-bookworm

ARG PROM_VERSION=2.53.0
ARG GRAFANA_VERSION=11.1.0
ARG ARCH=arm64

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_PORT=5002 \
    MLFLOW_TRACKING_URI=http://127.0.0.1:5002 \
    MODEL_NAME=HousePriceModel

WORKDIR /app

# System deps + uv + supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget curl tar ca-certificates libfontconfig1 supervisor procps \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Python deps first (cacheable)
COPY pyproject.toml uv.lock* ./
RUN uv pip install --system --no-cache-dir .

# App code
COPY . /app

# Data & config dirs
RUN mkdir -p /data/mlartifacts /etc/prometheus \
    /etc/grafana/provisioning/datasources /etc/grafana/provisioning/dashboards

# --- Prometheus (single binary) ---
RUN set -eux; cd /opt; \
    wget -q "https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-${ARCH}.tar.gz"; \
    tar -xzf "prometheus-${PROM_VERSION}.linux-${ARCH}.tar.gz"; \
    mv "prometheus-${PROM_VERSION}.linux-${ARCH}" prometheus; \
    rm "prometheus-${PROM_VERSION}.linux-${ARCH}.tar.gz"

# Use your prometheus.yml; patch target to local container API
COPY monitoring/prometheus.yml /etc/prometheus/prometheus.yml
RUN sed -i 's#host.docker.internal:8000#127.0.0.1:8000#g' /etc/prometheus/prometheus.yml

# --- Grafana (single binary) — robust extract (detect folder name) ---
RUN set -eux; cd /opt; \
    wget -q "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-${ARCH}.tar.gz"; \
    FOLDER="$(tar -tzf "grafana-${GRAFANA_VERSION}.linux-${ARCH}.tar.gz" | head -1 | cut -d/ -f1)"; \
    tar -xzf "grafana-${GRAFANA_VERSION}.linux-${ARCH}.tar.gz"; \
    mv "$FOLDER" grafana; \
    rm "grafana-${GRAFANA_VERSION}.linux-${ARCH}.tar.gz"

# Minimal grafana.ini
RUN printf '%s\n' \
  '[paths]' \
  'provisioning = /etc/grafana/provisioning' \
  '[server]' \
  'http_port = 3000' \
  '[security]' \
  'admin_user = admin' \
  'admin_password = admin' \
  > /etc/grafana/grafana.ini

# Provision datasource + dashboards
COPY monitoring/grafana/provisioning/datasources/datasource.yml /etc/grafana/provisioning/datasources/datasource.yml
COPY monitoring/grafana/provisioning/dashboards/ /etc/grafana/provisioning/dashboards/
RUN sed -i 's#http://prometheus-server-compose:9090#http://127.0.0.1:9090#g' /etc/grafana/provisioning/datasources/datasource.yml

# --- Supervisor configs (no heredocs) ---
COPY docker/supervisor/supervisord.conf /etc/supervisor/supervisord.conf
COPY docker/supervisor/stack.conf /etc/supervisor/conf.d/stack.conf

EXPOSE 8000 5002 9090 3000
VOLUME ["/data"]

CMD ["/usr/bin/supervisord","-c","/etc/supervisor/supervisord.conf"]
