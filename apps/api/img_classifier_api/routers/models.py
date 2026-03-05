"""Router for model management endpoints."""

import asyncio
import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..auth import require_api_key
from ..schemas import AvailableModelsResponse, LoadModelResponse, MessageResponse, ModelInfo


def _get_allowed_model_dirs() -> list[Path]:
    """Return the resolved list of directories from which models may be loaded."""
    dirs: list[Path] = [
        (Path.home() / ".local" / "share" / "img_classifier" / "models").resolve(),
        (Path.cwd() / "models").resolve(),
    ]
    env_dir = os.environ.get("IMG_CLASSIFIER_MODEL_DIR", "").strip()
    if env_dir:
        dirs.append(Path(env_dir).resolve())
    return dirs


class DiscoveredModelsResponse(BaseModel):
    """Response for discovered model files."""

    model_files: list[str]
    count: int


router = APIRouter()


@router.get("/models", response_model=AvailableModelsResponse)
async def list_models(request: Request) -> AvailableModelsResponse:
    """List all loaded models.

    Args:
        request: FastAPI request object

    Returns:
        AvailableModelsResponse: List of loaded models and current active model
    """
    model_manager = request.app.state.model_manager
    return AvailableModelsResponse(
        models=list(model_manager.model_info.values()),
        current_model=model_manager.current_model_name,
    )


@router.get("/models/discover", response_model=DiscoveredModelsResponse)
async def discover_models(request: Request) -> DiscoveredModelsResponse:
    """Re-scan for available model files in default directories.

    Args:
        request: FastAPI request object

    Returns:
        DiscoveredModelsResponse: List of discovered model file paths
    """
    model_manager = request.app.state.model_manager
    discovered = model_manager.discover_models()
    return DiscoveredModelsResponse(
        model_files=[str(path) for path in discovered],
        count=len(discovered),
    )


@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str, request: Request) -> ModelInfo:
    """Get information about a specific loaded model.

    Args:
        model_name: Name of the model
        request: FastAPI request object

    Returns:
        ModelInfo: Model information

    Raises:
        HTTPException: 404 if model not found
    """
    model_manager = request.app.state.model_manager
    try:
        return model_manager.get_info(model_name)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post(
    "/models/{model_name}/activate",
    dependencies=[Depends(require_api_key)],
    response_model=MessageResponse,
)
async def activate_model(model_name: str, request: Request) -> MessageResponse:
    """Set a model as the current active model.

    Args:
        model_name: Name of the model to activate
        request: FastAPI request object

    Returns:
        MessageResponse: Success message

    Raises:
        HTTPException: 404 if model not found
    """
    model_manager = request.app.state.model_manager
    try:
        model_manager.set_current_model(model_name)
        return MessageResponse(message=f"Model '{model_name}' is now active")
    except ValueError as e:
        raise HTTPException(404, str(e))


class LoadModelRequest(BaseModel):
    """Request body for loading a model."""

    model_path: str = Field(..., description="File path to the model")
    model_name: str | None = Field(None, description="Optional custom name for the model")


@router.post(
    "/models/load", dependencies=[Depends(require_api_key)], response_model=LoadModelResponse
)
async def load_model_endpoint(
    request_body: LoadModelRequest, request: Request
) -> LoadModelResponse:
    """Load a model from a file path.

    The supplied path must resolve to a location inside one of the configured
    allowed model directories (``IMG_CLASSIFIER_MODEL_DIR`` env var or the
    built-in defaults).  Paths outside those directories are rejected with 403
    to prevent path-traversal attacks.

    Args:
        request_body: LoadModelRequest with model_path and optional model_name
        request: FastAPI request object

    Returns:
        dict: Success message with loaded model name

    Raises:
        HTTPException: 403 if path not allowed, 404 if file not found, 500 on load error
    """
    model_manager = request.app.state.model_manager
    try:
        safe_path = Path(request_body.model_path).resolve()
        allowed_dirs = _get_allowed_model_dirs()
        if not any(safe_path.is_relative_to(allowed) for allowed in allowed_dirs):
            raise HTTPException(
                403,
                "Model path is not within an allowed directory. "
                f"Allowed locations: {[str(d) for d in allowed_dirs]}",
            )
        loaded_name = await asyncio.to_thread(
            model_manager.load_model, safe_path, request_body.model_name
        )
        return LoadModelResponse(message="Model loaded successfully", model_name=loaded_name)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error loading model: {str(e)}")
