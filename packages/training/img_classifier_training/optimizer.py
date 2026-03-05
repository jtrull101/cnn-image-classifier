"""Hyperparameter optimization for model training."""

import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from img_classifier_config import BaseConfig

logger = logging.getLogger(__name__)

# Default search space values
_DEFAULT_LEARNING_RATES: list[float] = [0.0001, 0.0005, 0.001, 0.005, 0.01]
_DEFAULT_BATCH_SIZES: list[int] = [16, 32, 64]
_DEFAULT_DROPOUT_RATES: list[float] = [0.2, 0.3, 0.4, 0.5]
_DEFAULT_ARCHITECTURES: list[str] = ["simple", "medium", "deep"]
_DEFAULT_NUM_EPOCHS: list[int] = [20, 30, 40, 50]

# Quick (reduced) search space values
_QUICK_LEARNING_RATES: list[float] = [0.001, 0.01]
_QUICK_BATCH_SIZES: list[int] = [32, 64]
_QUICK_DROPOUT_RATES: list[float] = [0.3, 0.4]
_QUICK_ARCHITECTURES: list[str] = ["simple", "medium"]
_QUICK_NUM_EPOCHS: list[int] = [20, 30]


class OptimizerType(StrEnum):
    """Hyperparameter optimizer algorithm types."""

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


@dataclass
class HyperparameterSpace:
    """Defines the search space for hyperparameters."""

    learning_rates: list[float]
    batch_sizes: list[int]
    dropout_rates: list[float]
    architectures: list[str]  # 'simple', 'medium', 'deep'
    num_epochs: list[int]

    @classmethod
    def default(cls) -> "HyperparameterSpace":
        """Create default hyperparameter search space."""
        return cls(
            learning_rates=_DEFAULT_LEARNING_RATES,
            batch_sizes=_DEFAULT_BATCH_SIZES,
            dropout_rates=_DEFAULT_DROPOUT_RATES,
            architectures=_DEFAULT_ARCHITECTURES,
            num_epochs=_DEFAULT_NUM_EPOCHS,
        )

    @classmethod
    def quick(cls) -> "HyperparameterSpace":
        """Create a smaller search space for quick experiments."""
        return cls(
            learning_rates=_QUICK_LEARNING_RATES,
            batch_sizes=_QUICK_BATCH_SIZES,
            dropout_rates=_QUICK_DROPOUT_RATES,
            architectures=_QUICK_ARCHITECTURES,
            num_epochs=_QUICK_NUM_EPOCHS,
        )


@dataclass
class TrialResult:
    """Result from a single hyperparameter trial."""

    trial_id: int
    hyperparameters: dict[str, Any]
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_loss: float
    val_loss: float
    test_loss: float
    training_time: float
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class HyperparameterOptimizer(ABC):
    """Base class for hyperparameter optimization strategies."""

    def __init__(
        self,
        config: BaseConfig,
        search_space: HyperparameterSpace,
        results_dir: Path | None = None,
        max_trials: int = 50,
        early_stop_patience: int = 10,
        target_accuracy: float = 0.95,
    ):
        """Initialize optimizer.

        Args:
            config: Base configuration
            search_space: Hyperparameter search space
            results_dir: Directory to save results
            max_trials: Maximum number of trials to run
            early_stop_patience: Stop if no improvement after N trials
            target_accuracy: Stop if this accuracy is reached
        """
        self.config = config
        self.search_space = search_space
        self.results_dir = results_dir or config.logs_dir / "optimization"
        self.max_trials = max_trials
        self.early_stop_patience = early_stop_patience
        self.target_accuracy = target_accuracy

        self.results: list[TrialResult] = []
        self.best_result: TrialResult | None = None
        self.trials_without_improvement = 0

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def suggest_hyperparameters(self, trial_number: int) -> dict[str, Any]:
        """Suggest next set of hyperparameters to try.

        Args:
            trial_number: Current trial number

        Returns:
            Dictionary of hyperparameter values
        """
        pass

    def should_stop(self) -> bool:
        """Check if optimization should stop.

        Returns:
            True if optimization should stop
        """
        # Stop if max trials reached
        if len(self.results) >= self.max_trials:
            return True

        # Stop if target accuracy reached
        if self.best_result and self.best_result.val_accuracy >= self.target_accuracy:
            return True

        # Stop if no improvement for a while
        if self.trials_without_improvement >= self.early_stop_patience:
            return True

        return False

    def record_result(self, result: TrialResult) -> None:
        """Record a trial result.

        Args:
            result: Trial result to record
        """
        self.results.append(result)

        # Update best result
        if self.best_result is None or result.val_accuracy > self.best_result.val_accuracy:
            self.best_result = result
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1

        # Save results
        self._save_results()

    def _save_results(self) -> None:
        """Save all results to JSON file."""
        results_file = self.results_dir / "optimization_results.json"

        data = {
            "search_space": asdict(self.search_space),
            "config": {
                "max_trials": self.max_trials,
                "target_accuracy": self.target_accuracy,
                "early_stop_patience": self.early_stop_patience,
            },
            "results": [r.to_dict() for r in self.results],
            "best_result": self.best_result.to_dict() if self.best_result else None,
        }

        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_summary(self) -> str:
        """Get optimization summary.

        Returns:
            Summary string
        """
        if not self.results:
            return "No trials completed yet."

        summary = []
        summary.append(f"\n{'=' * 60}")
        summary.append("Hyperparameter Optimization Summary")
        summary.append(f"{'=' * 60}")
        summary.append(f"Total trials: {len(self.results)}")
        summary.append(f"Best validation accuracy: {self.best_result.val_accuracy:.4f}")  # type: ignore
        summary.append(f"Best test accuracy: {self.best_result.test_accuracy:.4f}")  # type: ignore
        summary.append("\nBest hyperparameters:")
        for key, value in self.best_result.hyperparameters.items():  # type: ignore
            summary.append(f"  {key}: {value}")
        summary.append(f"{'=' * 60}\n")

        return "\n".join(summary)


class GridSearchOptimizer(HyperparameterOptimizer):
    """Grid search over all hyperparameter combinations."""

    def __init__(self, *args, **kwargs):
        """Initialize grid search optimizer."""
        super().__init__(*args, **kwargs)
        self._grid = self._create_grid()
        self._current_index = 0

    def _create_grid(self) -> list[dict[str, Any]]:
        """Create grid of all hyperparameter combinations.

        Returns:
            List of hyperparameter dictionaries
        """
        grid = []
        for lr in self.search_space.learning_rates:
            for bs in self.search_space.batch_sizes:
                for dr in self.search_space.dropout_rates:
                    for arch in self.search_space.architectures:
                        for epochs in self.search_space.num_epochs:
                            grid.append(
                                {
                                    "learning_rate": lr,
                                    "batch_size": bs,
                                    "dropout_rate": dr,
                                    "architecture": arch,
                                    "num_epochs": epochs,
                                }
                            )

        # Shuffle grid for randomization
        random.shuffle(grid)
        return grid

    def suggest_hyperparameters(self, trial_number: int) -> dict[str, Any]:
        """Suggest next hyperparameters from grid.

        Args:
            trial_number: Current trial number

        Returns:
            Dictionary of hyperparameter values
        """
        if self._current_index >= len(self._grid):
            raise StopIteration("Grid search completed")

        params = self._grid[self._current_index]
        self._current_index += 1
        return params


class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search over hyperparameter space."""

    def suggest_hyperparameters(self, trial_number: int) -> dict[str, Any]:
        """Randomly suggest hyperparameters.

        Args:
            trial_number: Current trial number

        Returns:
            Dictionary of hyperparameter values
        """
        return {
            "learning_rate": random.choice(self.search_space.learning_rates),
            "batch_size": random.choice(self.search_space.batch_sizes),
            "dropout_rate": random.choice(self.search_space.dropout_rates),
            "architecture": random.choice(self.search_space.architectures),
            "num_epochs": random.choice(self.search_space.num_epochs),
        }


class BayesianOptimizer(HyperparameterOptimizer):
    """Bayesian optimization using Gaussian processes (requires optuna)."""

    def __init__(self, *args, use_optuna: bool = True, **kwargs):
        """Initialize Bayesian optimizer.

        Args:
            use_optuna: Whether to use Optuna library
            *args: Positional arguments for parent class
            **kwargs: Keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)
        self.use_optuna = use_optuna
        self.study = None

        if use_optuna:
            try:
                import optuna  # type: ignore[import-not-found]

                self.study = optuna.create_study(
                    direction="maximize",
                    study_name=f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
            except ImportError:
                logger.warning("optuna not installed, falling back to random search")
                self.use_optuna = False

    def suggest_hyperparameters(self, trial_number: int) -> dict[str, Any]:
        """Suggest hyperparameters using Bayesian optimization.

        Args:
            trial_number: Current trial number

        Returns:
            Dictionary of hyperparameter values
        """
        if not self.use_optuna or self.study is None:
            # Fallback to random search
            return super().suggest_hyperparameters(trial_number)

        # Get suggestion from study (optuna already imported in __init__)
        trial = self.study.ask()
        params = {
            "learning_rate": trial.suggest_categorical(
                "learning_rate", self.search_space.learning_rates
            ),
            "batch_size": trial.suggest_categorical("batch_size", self.search_space.batch_sizes),
            "dropout_rate": trial.suggest_categorical(
                "dropout_rate", self.search_space.dropout_rates
            ),
            "architecture": trial.suggest_categorical(
                "architecture", self.search_space.architectures
            ),
            "num_epochs": trial.suggest_categorical("num_epochs", self.search_space.num_epochs),
        }

        return params


def create_optimizer(
    optimizer_type: OptimizerType,
    config: BaseConfig,
    search_space: HyperparameterSpace | None = None,
    **kwargs,
) -> HyperparameterOptimizer:
    """Factory function to create optimizer.

    Args:
        optimizer_type: Type of optimizer ('grid', 'random', 'bayesian')
        config: Base configuration
        search_space: Hyperparameter search space (default if None)
        **kwargs: Additional arguments for optimizer

    Returns:
        HyperparameterOptimizer instance
    """
    if search_space is None:
        search_space = HyperparameterSpace.default()

    optimizer_type = OptimizerType(str(optimizer_type).lower())
    if optimizer_type == OptimizerType.GRID:
        return GridSearchOptimizer(config, search_space, **kwargs)
    elif optimizer_type == OptimizerType.RANDOM:
        return RandomSearchOptimizer(config, search_space, **kwargs)
    elif optimizer_type == OptimizerType.BAYESIAN:
        return BayesianOptimizer(config, search_space, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
