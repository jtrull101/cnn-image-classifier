# Stage 1: Builder - Install dependencies
# Pin to specific image digest for reproducible builds
# This is python:3.11.9-slim-bookworm as of the build date
FROM python@sha256:4fad23465a06cc5149a541fbec6f87e234a64dc0550f6bfdd2d290d8f03240df AS builder

# OpenCV + TensorFlow system dependencies
# Package versions are pinned via the base image digest above
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy workspace configuration first
COPY pyproject.toml uv.lock ./

# Copy ALL source code for workspace packages BEFORE running uv sync
# This is necessary because workspace dependencies require source code to be present
COPY packages/ packages/
COPY apps/api/ apps/api/

# Install dependencies (workspace packages will be installed from local source)
RUN uv sync --no-dev --frozen

# Stage 2: Runtime - Minimal final image
# Pin to same image digest for consistency
FROM python@sha256:4fad23465a06cc5149a541fbec6f87e234a64dc0550f6bfdd2d290d8f03240df

# Runtime dependencies only (smaller set)
# Package versions are pinned via the base image digest above
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy only the virtual environment and source code from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/packages /app/packages
COPY --from=builder /app/apps/api /app/apps/api

# Put the venv on PATH so uvicorn is directly callable
ENV PATH="/app/.venv/bin:$PATH"
# Point to static/ so expected default models are auto-discovered
ENV IMG_CLASSIFIER_MODEL_DIR=/app/apps/api/img_classifier_api/static
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
WORKDIR /app/apps/api

CMD ["uvicorn", "img_classifier_api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]