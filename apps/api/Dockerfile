# Stage 1: Builder - Install dependencies
# Pin to specific Python 3.11 version with digest for reproducibility
FROM python:3.11.9-slim-bookworm@sha256:4a8b0e8cf2e30a68eac4db5fc639c203d67e0b06b6c16d5c82b17b4f6c1e59f1 AS builder

# OpenCV + TensorFlow system dependencies
# Versions pinned for Debian Bookworm (Python 3.11 base)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0=2.74.6-2+deb12u3 \
    libsm6=2:1.2.3-1 \
    libxext6=2:1.3.4-1+b1 \
    libxrender-dev=1:0.9.10-1.1 \
    libgomp1=12.2.0-14 \
    libgl1=1.6.0-1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./
COPY packages/config/pyproject.toml packages/config/
COPY packages/utils/pyproject.toml packages/utils/
COPY packages/data/pyproject.toml packages/data/
COPY packages/models/pyproject.toml packages/models/
COPY packages/training/pyproject.toml packages/training/
COPY packages/cli/pyproject.toml packages/cli/
COPY apps/api/pyproject.toml apps/api/

# Install dependencies only
RUN uv sync --no-dev --frozen

# Copy source code after dependencies are installed
COPY packages/ packages/
COPY apps/api/ apps/api/

# Stage 2: Runtime - Minimal final image
# Pin to same Python 3.11 version as builder for consistency
FROM python:3.11.9-slim-bookworm@sha256:4a8b0e8cf2e30a68eac4db5fc639c203d67e0b06b6c16d5c82b17b4f6c1e59f1

# Runtime dependencies only (smaller set)
# Versions pinned for Debian Bookworm (Python 3.11 base)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0=2.74.6-2+deb12u3 \
    libsm6=2:1.2.3-1 \
    libxext6=2:1.3.4-1+b1 \
    libgomp1=12.2.0-14 \
    libgl1=1.6.0-1 \
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
