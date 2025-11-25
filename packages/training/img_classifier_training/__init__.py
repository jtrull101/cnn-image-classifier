"""Training pipeline package for Alzheimer's MRI CNN."""

from .callbacks import AccuracyThresholdCallback
from .trainer import Trainer
from .optimizer import (
    HyperparameterOptimizer,
    HyperparameterSpace,
    TrialResult,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    create_optimizer,
)
from .orchestrator import TrainingOrchestrator

__all__ = [
    "AccuracyThresholdCallback",
    "Trainer",
    "HyperparameterOptimizer",
    "HyperparameterSpace",
    "TrialResult",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "BayesianOptimizer",
    "create_optimizer",
    "TrainingOrchestrator",
]
__version__ = "0.1.0"
