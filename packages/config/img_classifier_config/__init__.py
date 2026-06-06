"""Configuration package for image classification."""

from img_classifier_config.base_config import BaseConfig
from img_classifier_config.dataset_config import (
    ArchitectureComplexity,
    DatasetConfig,
    DatasetDetector,
)


__all__ = [
    "ArchitectureComplexity",
    "BaseConfig",
    "DatasetConfig",
    "DatasetDetector",
]
__version__ = "0.1.0"
