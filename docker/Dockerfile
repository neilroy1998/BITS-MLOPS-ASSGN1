# Stage 1: Use a Python base image that matches your project's requirements (>=3.13)
# Note: As of now, 3.13 is a release candidate, which is fine for this project.
FROM python:3.13-rc-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Install uv, our package manager
RUN pip install uv

# Copy the files that define the project and its dependencies.
# Copying the lock file ensures reproducible builds if it exists.
COPY pyproject.toml uv.lock* ./

# Install production dependencies using uv.
# The "." tells uv to install from the pyproject.toml in the current directory.
# We use --system to install into the global site-packages, a common container practice.
RUN uv pip install --system --no-cache-dir .

# Copy the rest of the application code into the container
COPY . .

# Expose the port the API will run on
EXPOSE 8000

# Command to run the application using uvicorn
# 0.0.0.0 makes the server accessible from outside the container
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
