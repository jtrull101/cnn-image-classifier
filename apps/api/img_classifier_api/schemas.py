"""Shared Pydantic models for API requests and responses."""

from typing import List, Optional

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


class PredictionResponse(BaseModel):
    """Response from a prediction request."""

    class_name: str = Field(..., description="The predicted class label")
    confidence: float = Field(..., description="Confidence score for the prediction")
    probabilities: dict[str, float] = Field(..., description="Probability for each class")
    model_name: str = Field(..., description="Name of the model used")
