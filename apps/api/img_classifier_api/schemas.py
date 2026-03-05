"""Shared Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """Generic single-message response."""

    message: str


class LoadModelResponse(BaseModel):
    """Response after loading a model."""

    message: str
    model_name: str


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: int
    current_model: str | None
    websocket_connections: int


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    name: str
    path: str
    num_classes: int
    class_names: list[str]
    input_shape: list[int]
    accuracy: float | None = None


class AvailableModelsResponse(BaseModel):
    """Response listing available models."""

    models: list[ModelInfo]
    current_model: str | None = None


class PredictionResponse(BaseModel):
    """Response from a prediction request."""

    class_name: str = Field(..., description="The predicted class label")
    confidence: float = Field(..., description="Confidence score for the prediction")
    probabilities: dict[str, float] = Field(..., description="Probability for each class")
    model_name: str = Field(..., description="Name of the model used")
