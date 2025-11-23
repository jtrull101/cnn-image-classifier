"""Neural network models package for Alzheimer's MRI CNN."""

from .base_model import BaseModel
from .cnn_classifier import CNNClassifier, SimpleCNN

__all__ = ["BaseModel", "CNNClassifier", "SimpleCNN"]
__version__ = "0.1.0"
