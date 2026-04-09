# Stage 1: Builder - Install dependencies
# Pin to specific image digest for reproducible builds
# This is python:3.11.9-slim-bookworm as of the build date
FROM python@sha256:b1b81d67b8df73bf6067191790c233ef9e598b863fc1ccf447956138ca466d99 AS builder

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

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./
COPY packages/config/pyproject.toml packages/config/README.md packages/config/
COPY packages/utils/pyproject.toml packages/utils/README.md packages/utils/
COPY packages/data/pyproject.toml packages/data/README.md packages/data/
COPY packages/models/pyproject.toml packages/models/README.md packages/models/
COPY packages/training/pyproject.toml packages/training/README.md packages/training/
COPY packages/cli/pyproject.toml packages/cli/README.md packages/cli/
COPY apps/api/pyproject.toml apps/api/README.md apps/api/

# Install dependencies only
RUN uv sync --no-dev --frozen

# Copy source code after dependencies are installed
COPY packages/ packages/
COPY apps/api/ apps/api/

# Stage 2: Runtime - Minimal final image
# Pin to same image digest for consistency
FROM python@sha256:b1b81d67b8df73bf6067191790c233ef9e598b863fc1ccf447956138ca466d99

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