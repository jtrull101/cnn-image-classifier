"""Training orchestrator for coordinating the complete ML pipeline."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from img_classifier_config import BaseConfig, DatasetConfig, DatasetDetector
from img_classifier_data import BaseDataLoader, ImageDataLoader
from img_classifier_models import ArchitectureFactory, BaseModel

from .optimizer import (
    HyperparameterOptimizer,
    HyperparameterSpace,
    TrialResult,
    create_optimizer,
)
from .trainer import Trainer

import tensorflow as tf

logger = logging.getLogger(__name__)


class _DynamicModel(BaseModel):
    """Thin BaseModel wrapper around an already-built Keras model."""

    def __init__(self, config: BaseConfig, keras_model: tf.keras.Model):
        super().__init__(config)
        self.model = keras_model

    def build(self) -> tf.keras.Model:
        return self.model  # type: ignore[return-value]


class TrainingOrchestrator:
    """Orchestrates the complete training pipeline.

    This class coordinates:
    1. Dataset detection and configuration
    2. Architecture generation
    3. Hyperparameter optimization
    4. Model training and evaluation
    5. Results logging and model saving
    """

    def __init__(
        self,
        config: BaseConfig,
        data_loader: BaseDataLoader | None = None,
        optimize_hyperparameters: bool = False,
        optimizer_type: Literal["grid", "random", "bayesian"] = "random",
        search_space: HyperparameterSpace | None = None,
        max_trials: int = 20,
    ):
        """Initialize the orchestrator.

        Args:
            config: Base configuration
            data_loader: Data loader (will create if None)
            optimize_hyperparameters: Whether to run hyperparameter optimization
            optimizer_type: Type of optimizer ('grid', 'random', 'bayesian')
            search_space: Hyperparameter search space
            max_trials: Maximum number of optimization trials
        """
        self.config = config
        self.data_loader = data_loader or ImageDataLoader(config)
        self.optimize_hyperparameters = optimize_hyperparameters
        self.optimizer_type = optimizer_type
        self.search_space = search_space
        self.max_trials = max_trials

        self.optimizer: HyperparameterOptimizer | None = None
        self.best_model: BaseModel | None = None
        self.best_config: BaseConfig | None = None

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: Path,
        project_name: str | None = None,
        working_dir: Path | None = None,
        **kwargs,
    ) -> "TrainingOrchestrator":
        """Create orchestrator by auto-detecting dataset.

        Args:
            dataset_path: Path to dataset directory
            project_name: Name for the project
            working_dir: Working directory for outputs
            **kwargs: Additional arguments for orchestrator

        Returns:
            TrainingOrchestrator instance
        """
        detector = DatasetDetector(dataset_path)
        config = detector.create_config(
            project_name=project_name,
            working_dir=working_dir,
        )

        info = detector.detect()
        logger.info("Dataset Auto-Detection Results")
        logger.info("Dataset: %s", info["dataset_path"])
        logger.info("Classes: %d - %s", info["num_classes"], ", ".join(info["class_names"]))
        logger.info("Total images: %d", info["total_images"])
        logger.info("Balanced: %s", info["is_balanced"])
        logger.info("Recommended complexity: %s", info["recommended_complexity"])

        return cls(config, **kwargs)

    def run(self, plot: bool = False) -> dict:
        """Run the complete training pipeline.

        Args:
            plot: Whether to plot training history

        Returns:
            Dictionary containing model, history, and metrics
        """
        logger.info("Starting Training Orchestrator")
        logger.info("Project: %s", self.config.project_name)
        logger.info("Dataset: %s", self.config.dataset_name)
        logger.info("Classes: %d", self.config.num_classes)
        logger.info("Optimization: %s", "Enabled" if self.optimize_hyperparameters else "Disabled")

        # Prepare dataset
        if not self._prepare_dataset():
            raise RuntimeError("Failed to prepare dataset")

        # Run optimization or single training
        if self.optimize_hyperparameters:
            return self._run_optimization(plot=plot)
        return self._run_single_training(plot=plot)

    def _prepare_dataset(self) -> bool:
        """Prepare the dataset for training.

        Returns:
            True if successful
        """
        logger.info("Preparing dataset...")

        # Download if needed
        if hasattr(self.data_loader, "download_dataset"):
            self.data_loader.download_dataset()  # type: ignore

        # Prepare/extract if needed
        if hasattr(self.data_loader, "prepare_dataset"):
            return self.data_loader.prepare_dataset()  # type: ignore

        return True

    def _run_single_training(self, plot: bool = False) -> dict:
        """Run a single training session.

        Args:
            plot: Whether to plot training history
        """
        logger.info("Running single training session...")

        # Create model from architecture factory
        model_keras = ArchitectureFactory.create(self.config)
        model = _DynamicModel(self.config, model_keras)
        model.compile()

        # Create trainer and run
        trainer = Trainer(self.config, model, self.data_loader)
        acc, loss = trainer.run(plot=plot, force_save=True)

        self.best_model = model
        self.best_config = self.config

        logger.info("Training completed! Final accuracy: %.4f  Final loss: %.4f", acc, loss)
        return {
            "model": model,
            "history": trainer.history,
            "metrics": {"accuracy": acc, "loss": loss},
        }

    def _run_optimization(self, plot: bool = False) -> dict:
        """Run hyperparameter optimization.

        Args:
            plot: Whether to plot training history
        """
        logger.info(
            "Starting hyperparameter optimization (%s). Max trials: %d",
            self.optimizer_type,
            self.max_trials,
        )

        # Create optimizer
        self.optimizer = create_optimizer(
            self.optimizer_type,
            self.config,
            self.search_space,
            max_trials=self.max_trials,
        )

        trial_number = 0

        while not self.optimizer.should_stop():
            trial_number += 1

            # Get hyperparameters for this trial
            hyperparams = self.optimizer.suggest_hyperparameters(trial_number)

            logger.info(
                "Trial %d/%d — hyperparameters: %s", trial_number, self.max_trials, hyperparams
            )

            # Create config for this trial
            trial_config = self._create_trial_config(hyperparams)

            # Run training trial
            result = self._run_trial(trial_config, trial_number, plot=plot)

            # Record result
            self.optimizer.record_result(result)

            logger.info(
                "Trial %d completed — val_acc: %.4f  test_acc: %.4f  time: %.1fs",
                trial_number,
                result.val_accuracy,
                result.test_accuracy,
                result.training_time,
            )
            if self.optimizer.best_result:
                logger.info("Best so far: %.4f", self.optimizer.best_result.val_accuracy)

        logger.info("%s", self.optimizer.get_summary())

        metrics = {}
        if self.optimizer and self.optimizer.best_result:
            best = self.optimizer.best_result
            metrics = {
                "train_accuracy": best.train_accuracy,
                "val_accuracy": best.val_accuracy,
                "test_accuracy": best.test_accuracy,
                "train_loss": best.train_loss,
                "val_loss": best.val_loss,
                "test_loss": best.test_loss,
            }

        return {"model": self.best_model, "history": None, "metrics": metrics}

    def _create_trial_config(self, hyperparams: dict) -> BaseConfig:
        """Create configuration for a trial.

        Args:
            hyperparams: Hyperparameter dictionary

        Returns:
            Configuration for this trial
        """
        # Create a copy of the base config
        config_dict = self.config.model_dump()

        # Update with trial hyperparameters
        config_dict["learning_rate"] = hyperparams["learning_rate"]
        config_dict["batch_size"] = hyperparams["batch_size"]
        config_dict["dropout_rate"] = hyperparams["dropout_rate"]
        config_dict["num_epochs"] = hyperparams["num_epochs"]
        config_dict["architecture_complexity"] = hyperparams["architecture"]

        # Create new config instance
        if isinstance(self.config, DatasetConfig):
            return DatasetConfig(**config_dict)
        else:
            return BaseConfig(**config_dict)

    def _run_trial(
        self,
        trial_config: BaseConfig,
        trial_number: int,
        plot: bool = False,
    ) -> TrialResult:
        """Run a single optimization trial.

        Args:
            trial_config: Configuration for this trial
            trial_number: Trial number
            plot: Whether to plot training history

        Returns:
            TrialResult with metrics
        """
        start_time = time.time()

        try:
            # Create model from architecture factory
            model_keras = ArchitectureFactory.create(trial_config)
            model = _DynamicModel(trial_config, model_keras)
            model.compile()

            # Create trainer and run
            trainer = Trainer(trial_config, model, self.data_loader)

            # Prepare data once if not already done
            x_train, y_train, x_val, y_val, x_test, y_test = trainer.prepare_data()

            # Train
            history = trainer.train(x_train, y_train, x_val, y_val)

            # Evaluate
            test_loss, test_acc = trainer.evaluate(x_test, y_test)

            # Get training metrics
            train_acc = history.history["accuracy"][-1]
            train_loss = history.history["loss"][-1]
            val_acc = history.history["val_accuracy"][-1]
            val_loss = history.history["val_loss"][-1]

            training_time = time.time() - start_time

            # Save model if it's the best so far
            if self.optimizer and (
                self.optimizer.best_result is None
                or val_acc > self.optimizer.best_result.val_accuracy
            ):
                self.best_model = model
                self.best_config = trial_config

                # Save to disk
                model_path = trial_config.models_dir / f"best_model_trial_{trial_number}.keras"
                model.save(model_path)

            # Clean up
            trainer.cleanup()

            return TrialResult(
                trial_id=trial_number,
                hyperparameters={
                    "learning_rate": trial_config.learning_rate,
                    "batch_size": trial_config.batch_size,
                    "dropout_rate": trial_config.dropout_rate,
                    "architecture": trial_config.architecture_complexity
                    if hasattr(trial_config, "architecture_complexity")
                    else "unknown",  # type: ignore
                    "num_epochs": trial_config.num_epochs,
                },
                train_accuracy=float(train_acc),
                val_accuracy=float(val_acc),
                test_accuracy=float(test_acc),
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                test_loss=float(test_loss),
                training_time=training_time,
                timestamp=datetime.now().isoformat(),
            )

        except Exception:
            logger.exception("Trial %d failed", trial_number)
            # Return a failed result
            return TrialResult(
                trial_id=trial_number,
                hyperparameters={
                    "learning_rate": trial_config.learning_rate,
                    "batch_size": trial_config.batch_size,
                    "dropout_rate": trial_config.dropout_rate,
                    "architecture": trial_config.architecture_complexity
                    if hasattr(trial_config, "architecture_complexity")
                    else "unknown",  # type: ignore
                    "num_epochs": trial_config.num_epochs,
                },
                train_accuracy=0.0,
                val_accuracy=0.0,
                test_accuracy=0.0,
                train_loss=999.0,
                val_loss=999.0,
                test_loss=999.0,
                training_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
            )
