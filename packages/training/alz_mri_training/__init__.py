"""Training pipeline package for Alzheimer's MRI CNN."""

from .callbacks import AccuracyThresholdCallback
from .trainer import Trainer

__all__ = ["AccuracyThresholdCallback", "Trainer"]
__version__ = "0.1.0"
