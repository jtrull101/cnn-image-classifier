FROM python:3.11-slim

# OpenCV + TensorFlow system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/api/ apps/api/

RUN uv sync --no-dev

# Put the venv on PATH so uvicorn is directly callable
ENV PATH="/app/.venv/bin:$PATH"
# Point to static/ so optimal_weights_98%.keras is auto-discovered
ENV IMG_CLASSIFIER_MODEL_DIR=/app/apps/api/img_classifier_api/static
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
WORKDIR /app/apps/api

CMD ["uvicorn", "img_classifier_api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
