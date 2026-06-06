"""FastAPI application for generic image classification.

This API provides endpoints for:
- Image classification predictions
- Model information and metadata
- Multiple model support
- Dynamic class label handling
"""

import json
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any

import tensorflow as tf
from fastapi import (
    Depends,
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from img_classifier_api.auth import require_api_key
from img_classifier_api.database import init_db
from img_classifier_api.routers import analytics, datasets, history, models, predictions, training
from img_classifier_api.schemas import HealthCheckResponse, ModelInfo
from img_classifier_api.websocket_manager import manager as ws_manager


logger = logging.getLogger(__name__)

# Setup templates and static files
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)


def _parse_accuracy_from_filename(filename: str) -> float | None:
    """Parse an accuracy percentage embedded in a model filename.

    Expects the pattern ``<name>_<integer>%<rest>``, e.g.
    ``model_name_95%_acc_...``.  Returns the value as a fraction
    (e.g. 0.95), or ``None`` if the pattern is not found.
    """
    if "%" not in filename:
        return None
    try:
        percent_pos = filename.find("%")
        before_percent = filename[:percent_pos]
        if "_" not in before_percent:
            return None
        acc_str = before_percent[before_percent.rfind("_") + 1 :]
        return float(acc_str) / 100.0
    except (ValueError, IndexError):
        return None


def _default_model_dirs() -> list[Path]:
    """Return the default model search directories, respecting env config."""
    _model_dir_env = os.environ.get("IMG_CLASSIFIER_MODEL_DIR")
    dirs = [
        Path(_model_dir_env).expanduser() if _model_dir_env else None,
        Path(
            os.environ.get(
                "IMG_CLASSIFIER_WORKING_DIR",
                str(Path.home() / ".local" / "share" / "img_classifier"),
            )
        )
        / "models",
        Path.cwd() / "models",
        Path(__file__).parent / "static",
    ]
    return [d for d in dirs if d is not None]


class ModelManager:
    """Manages multiple loaded models."""

    def __init__(self) -> None:
        """Initialize an empty model registry."""
        self.models: dict[str, tf.keras.Model] = {}
        self.model_info: dict[str, ModelInfo] = {}
        self.current_model_name: str | None = None
        self.default_model_dirs = _default_model_dirs()

    def discover_models(self) -> list[Path]:
        """Discover available model files."""
        found = []
        for model_dir in self.default_model_dirs:
            if model_dir.exists():
                found.extend(model_dir.glob("*.keras"))
                found.extend(model_dir.glob("*.h5"))
        return found

    def load_model(self, model_path: Path, model_name: str | None = None) -> str:
        """Load a model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if model_name is None:
            model_name = model_path.stem

        logger.info("Loading model '%s' from %s", model_name, model_path)
        # TF/Keras URL-decodes paths internally, so '%' in filenames causes a false
        # "file not found" even when the file exists. Copy to a temp path to work around it.
        tmp_dir: str | None = None
        load_path = model_path
        if "%" in model_path.name:
            tmp_dir = tempfile.mkdtemp()
            safe_name = model_path.name.replace("%", "_pct_")
            tmp_path = Path(tmp_dir) / safe_name
            shutil.copy2(model_path, tmp_path)
            load_path = tmp_path
        try:
            model = tf.keras.models.load_model(str(load_path))
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        input_shape = list(model.input_shape[1:])
        num_classes = int(model.output_shape[-1])

        class_names = [f"Class_{i}" for i in range(num_classes)]
        accuracy = None

        filename = model_path.name
        accuracy = _parse_accuracy_from_filename(filename)

        metadata_path = model_path.with_suffix(".json")
        if metadata_path.exists():
            try:
                with metadata_path.open("r") as f:
                    metadata = json.load(f)
                    class_names = metadata.get("class_names", class_names)
                    if "accuracy" in metadata:
                        accuracy = metadata["accuracy"]
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                logger.warning("Failed to load model metadata from %s", metadata_path)

        self.models[model_name] = model
        self.model_info[model_name] = ModelInfo(
            name=model_name,
            path=str(model_path),
            num_classes=num_classes,
            class_names=class_names,
            input_shape=input_shape,
            accuracy=accuracy,
        )

        if self.current_model_name is None:
            self.current_model_name = model_name

        return model_name

    def get_model(self, model_name: str | None = None) -> tf.keras.Model:
        """Get a loaded model."""
        if model_name is None:
            model_name = self.current_model_name

        if model_name is None:
            raise ValueError("No model loaded")

        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        return self.models[model_name]

    def get_info(self, model_name: str | None = None) -> ModelInfo:
        """Get model information."""
        if model_name is None:
            model_name = self.current_model_name

        if model_name is None:
            raise ValueError("No model loaded")

        if model_name not in self.model_info:
            raise ValueError(f"Model info not found: {model_name}")

        return self.model_info[model_name]

    def set_current_model(self, model_name: str) -> None:
        """Set the current active model."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        self.current_model_name = model_name

    def unload_model(self, model_name: str) -> None:
        """Remove a model from memory.

        If the unloaded model was the active model, the active model is reset to
        another loaded model (if any) or cleared entirely.

        Raises:
            ValueError: If the model is not currently loaded.
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        del self.models[model_name]
        del self.model_info[model_name]
        if self.current_model_name == model_name:
            self.current_model_name = next(iter(self.models), None)

    def auto_load_best_model(self) -> None:
        """Automatically load the best available model."""
        available = self.discover_models()
        if not available:
            logger.warning("No models found in default directories: %s", self.default_model_dirs)
            return

        best_model = None
        best_accuracy = -1

        for model_path in available:
            parsed = _parse_accuracy_from_filename(model_path.name)
            if parsed is not None:
                accuracy_pct = int(parsed * 100)
                if accuracy_pct > best_accuracy:
                    best_accuracy = accuracy_pct
                    best_model = model_path

        if best_model:
            self.load_model(best_model)
            logger.info("Auto-loaded best model with %d%% accuracy", best_accuracy)
        elif available:
            self.load_model(available[0])
            logger.info("Loaded first available model: %s", available[0].name)


class WebSocketMessageType(StrEnum):
    """WebSocket message type strings used in the /ws endpoint protocol."""

    PING = "ping"
    PONG = "pong"
    SUBSCRIBE = "subscribe"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBE = "unsubscribe"
    UNSUBSCRIBED = "unsubscribed"
    ECHO = "echo"


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class CSPMiddleware(BaseHTTPMiddleware):
    """Adds a Content-Security-Policy header to every HTML response."""

    _POLICY = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
        "https://cdn.tailwindcss.com https://unpkg.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self' ws: wss:; "
        "font-src 'self';"
    )

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Add security headers to HTML responses."""
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            response.headers["Content-Security-Policy"] = self._POLICY
        return response


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Lifespan context manager for startup and shutdown events."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    init_db()
    try:
        model_manager.auto_load_best_model()
    except Exception:
        logger.warning(
            "Could not auto-load model on startup; server will run without a model", exc_info=True
        )
    yield


app = FastAPI(
    title="Image Classification API",
    description="Generic CNN-based image classification system with multi-model support",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(CSPMiddleware)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.state.model_manager = model_manager
app.state.ws_manager = ws_manager

# All /api routes are protected by API key auth (disabled when IMG_CLASSIFIER_API_KEY is unset)
_api_deps = [Depends(require_api_key)]

app.include_router(models.router, prefix="/api", tags=["models"], dependencies=_api_deps)
app.include_router(predictions.router, prefix="/api", tags=["predictions"], dependencies=_api_deps)
app.include_router(history.router, prefix="/api", tags=["history"], dependencies=_api_deps)
app.include_router(analytics.router, prefix="/api", tags=["analytics"], dependencies=_api_deps)
app.include_router(training.router, prefix="/api", tags=["training"], dependencies=_api_deps)
app.include_router(datasets.router, prefix="/api", tags=["datasets"], dependencies=_api_deps)


# ---------------------------------------------------------------------------
# Web UI routes (no auth required)
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Serve the main web interface."""
    try:
        info = model_manager.get_info()
        context = {
            "model_name": info.name,
            "class_names": info.class_names,
            "accuracy": info.accuracy,
            "num_classes": info.num_classes,
        }
    except ValueError:
        context = {
            "model_name": None,
            "class_names": [],
            "accuracy": None,
            "num_classes": 0,
        }
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        models_loaded=len(model_manager.models),
        current_model=model_manager.current_model_name,
        websocket_connections=ws_manager.get_connection_count(),
    )


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request) -> HTMLResponse:
    """Serve the enhanced single-prediction interface."""
    return templates.TemplateResponse(request=request, name="predict.html", context={})


@app.get("/models-ui", response_class=HTMLResponse)
async def models_page(request: Request) -> HTMLResponse:
    """Serve the model management interface."""
    return templates.TemplateResponse(request=request, name="models.html", context={})


@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request) -> HTMLResponse:
    """Serve the multi-model comparison interface."""
    return templates.TemplateResponse(request=request, name="compare.html", context={})


@app.get("/datasets-ui", response_class=HTMLResponse)
async def datasets_page(request: Request) -> HTMLResponse:
    """Serve the dataset management interface."""
    return templates.TemplateResponse(request=request, name="datasets.html", context={})


@app.get("/train", response_class=HTMLResponse)
async def training_page(request: Request) -> HTMLResponse:
    """Serve the training interface."""
    return templates.TemplateResponse(request=request, name="training.html", context={})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time communication."""
    client_id = websocket.query_params.get("client_id")
    await ws_manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == WebSocketMessageType.PING:
                await ws_manager.send_personal_message(
                    {"type": WebSocketMessageType.PONG, "timestamp": data.get("timestamp")},
                    websocket,
                )
            elif message_type == WebSocketMessageType.SUBSCRIBE:
                room = data.get("room")
                if room:
                    ws_manager.join_room(websocket, room)
                    await ws_manager.send_personal_message(
                        {"type": WebSocketMessageType.SUBSCRIBED, "room": room}, websocket
                    )
            elif message_type == WebSocketMessageType.UNSUBSCRIBE:
                room = data.get("room")
                if room:
                    ws_manager.leave_room(websocket, room)
                    await ws_manager.send_personal_message(
                        {"type": WebSocketMessageType.UNSUBSCRIBED, "room": room}, websocket
                    )
            else:
                await ws_manager.send_personal_message(
                    {"type": WebSocketMessageType.ECHO, "data": data}, websocket
                )

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
