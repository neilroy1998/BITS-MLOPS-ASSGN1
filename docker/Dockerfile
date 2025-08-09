FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock* ./

RUN uv pip install --system --no-cache-dir .


COPY . .

EXPOSE 8000



CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
