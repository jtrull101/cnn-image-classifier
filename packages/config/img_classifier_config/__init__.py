"""Configuration package for Alzheimer's MRI CNN."""

from .base_config import BaseConfig
from .alzheimer_config import AlzheimerConfig
from .dataset_config import DatasetConfig, DatasetDetector

__all__ = ["BaseConfig", "AlzheimerConfig", "DatasetConfig", "DatasetDetector"]
__version__ = "0.1.0"
