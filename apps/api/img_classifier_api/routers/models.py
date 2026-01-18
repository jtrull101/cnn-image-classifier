"""Router for model management endpoints."""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


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


class DiscoveredModelsResponse(BaseModel):
    """Response for discovered model files."""

    model_files: List[str]
    count: int


router = APIRouter()


@router.get("/models", response_model=AvailableModelsResponse)
async def list_models(request: Request):
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
async def discover_models(request: Request):
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
async def get_model_info(model_name: str, request: Request):
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


@router.post("/models/{model_name}/activate")
async def activate_model(model_name: str, request: Request):
    """Set a model as the current active model.

    Args:
        model_name: Name of the model to activate
        request: FastAPI request object

    Returns:
        dict: Success message

    Raises:
        HTTPException: 404 if model not found
    """
    model_manager = request.app.state.model_manager
    try:
        model_manager.set_current_model(model_name)
        return {"message": f"Model '{model_name}' is now active"}
    except ValueError as e:
        raise HTTPException(404, str(e))


class LoadModelRequest(BaseModel):
    """Request body for loading a model."""

    model_path: str = Field(..., description="File path to the model")
    model_name: Optional[str] = Field(None, description="Optional custom name for the model")


@router.post("/models/load")
async def load_model_endpoint(request_body: LoadModelRequest, request: Request):
    """Load a model from a file path.

    Args:
        request_body: LoadModelRequest with model_path and optional model_name
        request: FastAPI request object

    Returns:
        dict: Success message with loaded model name

    Raises:
        HTTPException: 404 if file not found, 500 on load error
    """
    model_manager = request.app.state.model_manager
    try:
        path = Path(request_body.model_path)
        loaded_name = model_manager.load_model(path, request_body.model_name)
        return {"message": "Model loaded successfully", "model_name": loaded_name}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error loading model: {str(e)}")
