# Use the same base python image for consistency
FROM python:3.12-slim-bookworm

# Install mlflow
RUN pip install mlflow

# Expose the default mlflow port
EXPOSE 5000

# Command to run the mlflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "/mlruns", "--default-artifact-root", "/mlartifacts"]
