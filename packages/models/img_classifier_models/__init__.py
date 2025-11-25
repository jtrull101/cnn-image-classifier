"""Neural network models package for Alzheimer's MRI CNN."""

from .base_model import BaseModel
from .cnn_classifier import CnnClassifier, SimpleCnn
from .architecture_generator import (
    ArchitectureFactory,
    ArchitectureSpec,
    ArchitectureComplexity,
    ModelScaler,
    create_architecture_from_config,
)

__all__ = [
    "BaseModel",
    "CnnClassifier",
    "SimpleCnn",
    "ArchitectureFactory",
    "ArchitectureSpec",
    "ArchitectureComplexity",
    "ModelScaler",
    "create_architecture_from_config",
]
__version__ = "0.1.0"
