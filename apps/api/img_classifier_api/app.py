"""
FastAPI application for generic image classification.

This API provides endpoints for:
- Image classification predictions
- Model information and metadata
- Multiple model support
- Dynamic class label handling
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

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

from .auth import require_api_key
from .database import init_db
from .routers import analytics, history, models, predictions, training
from .schemas import ModelInfo
from .websocket_manager import manager as ws_manager

logger = logging.getLogger(__name__)

# Setup templates and static files
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)


def _default_model_dirs() -> List[Path]:
    """Return the default model search directories, respecting env config."""
    dirs = [
        Path(os.environ.get("IMG_CLASSIFIER_MODEL_DIR", "")).expanduser()
        if os.environ.get("IMG_CLASSIFIER_MODEL_DIR")
        else None,
        Path(
            os.environ.get(
                "IMG_CLASSIFIER_WORKING_DIR",
                os.path.join(Path.home(), ".local", "share", "img_classifier"),
            )
        )
        / "models",
        Path.cwd() / "models",
    ]
    return [d for d in dirs if d is not None]


class ModelManager:
    """Manages multiple loaded models."""

    def __init__(self):
        self.models: Dict[str, tf.keras.Model] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.current_model_name: Optional[str] = None
        self.default_model_dirs = _default_model_dirs()

    def discover_models(self) -> List[Path]:
        """Discover available model files."""
        found = []
        for model_dir in self.default_model_dirs:
            if model_dir.exists():
                found.extend(model_dir.glob("*.keras"))
                found.extend(model_dir.glob("*.h5"))
        return found

    def load_model(self, model_path: Path, model_name: Optional[str] = None) -> str:
        """Load a model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if model_name is None:
            model_name = model_path.stem

        logger.info("Loading model '%s' from %s", model_name, model_path)
        model = tf.keras.models.load_model(str(model_path))

        input_shape = list(model.input_shape[1:])
        num_classes = int(model.output_shape[-1])

        class_names = [f"Class_{i}" for i in range(num_classes)]
        accuracy = None

        filename = model_path.name
        if "%" in filename:
            try:
                percent_pos = filename.find("%")
                before_percent = filename[:percent_pos]
                if "_" in before_percent:
                    acc_str = before_percent[before_percent.rfind("_") + 1 :]
                    accuracy = float(acc_str) / 100.0
            except (ValueError, IndexError):
                pass

        metadata_path = model_path.with_suffix(".json")
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    class_names = metadata.get("class_names", class_names)
                    if "accuracy" in metadata:
                        accuracy = metadata["accuracy"]
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass

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

    def get_model(self, model_name: Optional[str] = None) -> tf.keras.Model:
        """Get a loaded model."""
        if model_name is None:
            model_name = self.current_model_name

        if model_name is None:
            raise ValueError("No model loaded")

        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        return self.models[model_name]

    def get_info(self, model_name: Optional[str] = None) -> ModelInfo:
        """Get model information."""
        if model_name is None:
            model_name = self.current_model_name

        if model_name is None:
            raise ValueError("No model loaded")

        if model_name not in self.model_info:
            raise ValueError(f"Model info not found: {model_name}")

        return self.model_info[model_name]

    def set_current_model(self, model_name: str):
        """Set the current active model."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        self.current_model_name = model_name

    def auto_load_best_model(self):
        """Automatically load the best available model."""
        available = self.discover_models()
        if not available:
            logger.warning("No models found in default directories: %s", self.default_model_dirs)
            return

        best_model = None
        best_accuracy = -1

        for model_path in available:
            filename = model_path.name
            if "%" in filename:
                try:
                    percent_pos = filename.find("%")
                    before_percent = filename[:percent_pos]
                    if "_" in before_percent:
                        acc_str = before_percent[before_percent.rfind("_") + 1 :]
                        accuracy = int(acc_str)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model_path
                except (ValueError, IndexError):
                    pass

        if best_model:
            self.load_model(best_model)
            logger.info("Auto-loaded best model with %d%% accuracy", best_accuracy)
        elif available:
            self.load_model(available[0])
            logger.info("Loaded first available model: %s", available[0].name)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class CSPMiddleware(BaseHTTPMiddleware):
    """Adds a Content-Security-Policy header to every HTML response."""

    _POLICY = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' "
        "https://cdn.tailwindcss.com https://unpkg.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self' ws: wss:; "
        "font-src 'self';"
    )

    async def dispatch(self, request: Request, call_next) -> Response:
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
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    init_db()
    model_manager.auto_load_best_model()
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


# ---------------------------------------------------------------------------
# Web UI routes (no auth required)
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.models),
        "current_model": model_manager.current_model_name,
        "websocket_connections": ws_manager.get_connection_count(),
    }


@app.get("/train", response_class=HTMLResponse)
async def training_page(request: Request):
    """Serve the training interface."""
    return templates.TemplateResponse(request=request, name="training.html", context={})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    client_id = websocket.query_params.get("client_id")
    await ws_manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "ping":
                await ws_manager.send_personal_message(
                    {"type": "pong", "timestamp": data.get("timestamp")}, websocket
                )
            elif message_type == "subscribe":
                room = data.get("room")
                if room:
                    ws_manager.join_room(websocket, room)
                    await ws_manager.send_personal_message(
                        {"type": "subscribed", "room": room}, websocket
                    )
            elif message_type == "unsubscribe":
                room = data.get("room")
                if room:
                    ws_manager.leave_room(websocket, room)
                    await ws_manager.send_personal_message(
                        {"type": "unsubscribed", "room": room}, websocket
                    )
            else:
                await ws_manager.send_personal_message({"type": "echo", "data": data}, websocket)

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
