"""
FastAPI application for generic image classification.

This API provides endpoints for:
- Image classification predictions
- Model information and metadata
- Multiple model support
- Dynamic class label handling
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tensorflow as tf
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .database import init_db
from .routers import analytics, history, models, predictions
from .websocket_manager import manager as ws_manager

# This will be defined after ModelManager is initialized
# Setup templates and static files
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)


# Pydantic models for request/response
# TODO extract to models class
class PredictionResponse(BaseModel):
    """Response model for predictions."""

    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    model_name: str = Field(..., description="Name of the model used")


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    name: str
    path: str
    num_classes: int
    class_names: List[str]
    input_shape: List[int]
    accuracy: Optional[float] = None


class AvailableModelsResponse(BaseModel):
    """Response listing available models."""

    models: List[ModelInfo]
    current_model: Optional[str] = None


# Global model cache
# TODO separate class?
class ModelManager:
    """Manages multiple loaded models."""

    def __init__(self):
        self.models: Dict[str, tf.keras.Model] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.current_model_name: Optional[str] = None
        self.default_model_dirs = [
            Path("/tmp/img_classifier/models"),
            Path("/tmp/img_classifier_cnn/models"),
            Path.cwd() / "models",
        ]

    def discover_models(self) -> List[Path]:
        """Discover available model files."""
        models = []
        for model_dir in self.default_model_dirs:
            if model_dir.exists():
                models.extend(model_dir.glob("*.keras"))
                models.extend(model_dir.glob("*.h5"))
        return models

    def load_model(self, model_path: Path, model_name: Optional[str] = None) -> str:
        """Load a model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Generate model name
        if model_name is None:
            model_name = model_path.stem

        # Load model
        print(f"Loading model: {model_name} from {model_path}")
        model = tf.keras.models.load_model(str(model_path))

        # Extract model info
        input_shape = list(model.input_shape[1:])
        num_classes = int(model.output_shape[-1])

        # Try to extract class names and accuracy from filename
        class_names = [f"Class_{i}" for i in range(num_classes)]
        accuracy = None

        filename = model_path.name
        if "%" in filename:
            try:
                # Find the percentage sign position
                percent_pos = filename.find("%")
                # Find the underscore before the percentage
                before_percent = filename[:percent_pos]
                if "_" in before_percent:
                    acc_str = before_percent[before_percent.rfind("_") + 1 :]
                    accuracy = float(acc_str) / 100.0
            except (ValueError, IndexError):
                pass

        # Check for metadata file
        metadata_path = model_path.with_suffix(".json")
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    class_names = metadata.get("class_names", class_names)
                    if "accuracy" in metadata:
                        accuracy = metadata["accuracy"]
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                # Metadata file doesn't exist or is invalid, use defaults
                pass

        # Store model and info
        self.models[model_name] = model
        self.model_info[model_name] = ModelInfo(
            name=model_name,
            path=str(model_path),
            num_classes=num_classes,
            class_names=class_names,
            input_shape=input_shape,
            accuracy=accuracy,
        )

        # Set as current if first model
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
        models = self.discover_models()
        if not models:
            print("Warning: No models found in default directories")
            return

        # Find model with highest accuracy in filename
        best_model = None
        best_accuracy = -1

        for model_path in models:
            filename = model_path.name
            if "%" in filename:
                try:
                    # Find the percentage sign position
                    percent_pos = filename.find("%")
                    # Find the underscore before the percentage
                    before_percent = filename[:percent_pos]
                    if "_" in before_percent:
                        acc_str = before_percent[before_percent.rfind("_") + 1 :]
                        accuracy = int(acc_str)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model_path
                except (ValueError, IndexError):
                    pass

        # Load best model or first available
        if best_model:
            self.load_model(best_model)
            print(f"Auto-loaded best model with {best_accuracy}% accuracy")
        elif models:
            self.load_model(models[0])
            print(f"Loaded first available model: {models[0].name}")


# Initialize model manager
model_manager = ModelManager()


# Define lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    init_db()  # Initialize database
    model_manager.auto_load_best_model()
    yield
    # Shutdown (if needed)
    # Add cleanup code here


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Image Classification API",
    description="Generic CNN-based image classification system with multi-model support",
    version="0.2.0",
    lifespan=lifespan,
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Store manager instances in app state for router access
app.state.model_manager = model_manager
app.state.ws_manager = ws_manager

# Include routers with /api prefix
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(predictions.router, prefix="/api", tags=["predictions"])
app.include_router(history.router, prefix="/api", tags=["history"])
app.include_router(analytics.router, prefix="/api", tags=["analytics"])


# API Endpoints


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    try:
        info = model_manager.get_info()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "model_name": info.name,
                "class_names": info.class_names,
                "accuracy": info.accuracy,
                "num_classes": info.num_classes,
            },
        )
    except ValueError:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "model_name": "No model loaded",
                "class_names": [],
                "accuracy": None,
                "num_classes": 0,
            },
        )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
):
    """
    Predict the class of an uploaded image.

    Args:
        file: Image file to classify
        model_name: Optional model name (uses current model if not specified)

    Returns:
        PredictionResponse with class prediction and confidence
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Get model and info
    try:
        model = model_manager.get_model(model_name)
        info = model_manager.get_info(model_name)
    except ValueError as e:
        raise HTTPException(404, str(e))

    # Read and process image
    try:
        # Read file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(400, "Unable to decode image")

        # Resize to model input shape
        input_shape = tuple(info.input_shape[:2])
        image_resized = cv2.resize(image, input_shape)

        # Normalize
        image_array = np.asarray(image_resized, dtype=np.float32) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

        # Predict
        predictions = model.predict(image_batch, verbose=0)
        probabilities = predictions[0]

        # Get top prediction
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = info.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # Build probability dict
        prob_dict = {
            class_name: float(prob) for class_name, prob in zip(info.class_names, probabilities)
        }

        return PredictionResponse(
            class_name=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            model_name=info.name,
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")


@app.get("/api/models", response_model=AvailableModelsResponse)
async def list_models():
    """List all available models."""
    return AvailableModelsResponse(
        models=list(model_manager.model_info.values()),
        current_model=model_manager.current_model_name,
    )


@app.get("/api/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    try:
        return model_manager.get_info(model_name)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/api/models/{model_name}/activate")
async def activate_model(model_name: str):
    """Set a model as the current active model."""
    try:
        model_manager.set_current_model(model_name)
        return {"message": f"Model '{model_name}' is now active"}
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/api/models/load")
async def load_model_endpoint(model_path: str, model_name: Optional[str] = None):
    """Load a model from a file path."""
    try:
        path = Path(model_path)
        loaded_name = model_manager.load_model(path, model_name)
        return {"message": "Model loaded successfully", "model_name": loaded_name}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error loading model: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.models),
        "current_model": model_manager.current_model_name,
        "websocket_connections": ws_manager.get_connection_count(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time communication.

    Handles:
    - Connection management
    - Ping/pong heartbeat
    - Real-time updates for predictions, model loading, batch processing
    """
    client_id = websocket.query_params.get("client_id")
    await ws_manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "ping":
                # Respond to heartbeat
                await ws_manager.send_personal_message(
                    {"type": "pong", "timestamp": data.get("timestamp")}, websocket
                )
            elif message_type == "subscribe":
                # Subscribe to specific room/channel
                room = data.get("room")
                if room:
                    ws_manager.join_room(websocket, room)
                    await ws_manager.send_personal_message(
                        {"type": "subscribed", "room": room}, websocket
                    )
            elif message_type == "unsubscribe":
                # Unsubscribe from room/channel
                room = data.get("room")
                if room:
                    ws_manager.leave_room(websocket, room)
                    await ws_manager.send_personal_message(
                        {"type": "unsubscribed", "room": room}, websocket
                    )
            else:
                # Echo unknown messages back (for debugging)
                await ws_manager.send_personal_message({"type": "echo", "data": data}, websocket)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
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
