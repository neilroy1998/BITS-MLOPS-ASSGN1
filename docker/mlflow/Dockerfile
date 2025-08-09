# docker/mlflow/Dockerfile

# Use the same base python image for consistency
FROM python:3.12-slim-bookworm

# Update package list and install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install mlflow and gunicorn for a more robust server
RUN pip install mlflow gunicorn

# Expose the correct port
EXPOSE 5002

# Command to run the mlflow server on port 5002
CMD ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5002", \
    "--backend-store-uri", "/mlruns", \
    "--artifacts-destination", "/mlartifacts", \
    "--default-artifact-root", "/mlartifacts"]
