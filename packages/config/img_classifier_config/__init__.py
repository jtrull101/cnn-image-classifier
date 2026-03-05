"""Configuration package for image classification."""

from .base_config import BaseConfig
from .dataset_config import ArchitectureComplexity, DatasetConfig, DatasetDetector

__all__ = [
    "BaseConfig",
    "ArchitectureComplexity",
    "DatasetConfig",
    "DatasetDetector",
    "ArchitectureComplexity",
]
__version__ = "0.1.0"
