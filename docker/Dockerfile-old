# Use the same Python base image
FROM python:3.12-slim-bookworm

# The most efficient way to install uv: copy the binary from the official image.
# This avoids needing curl and extra layers. We place it in /usr/local/bin which is already on the PATH.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set the working directory
WORKDIR /app

# Copy only the dependency files first to leverage Docker's layer caching.
# This layer will only be rebuilt if your dependencies change.
COPY pyproject.toml uv.lock ./

# Install dependencies using uv.
# The --system flag installs packages into the system's Python site-packages,
# which is standard practice for simple Docker containers.
RUN uv  sync

# Now copy the rest of your application code
COPY . .

# Expose the port your application will run on
EXPOSE 8000

# Set the command to run your application
# Using uvicorn directly is fine since it's now in the system's Python environment.
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
