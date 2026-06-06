"""Base model interface."""

import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

import tensorflow as tf

from img_classifier_config import BaseConfig


logger = logging.getLogger(__name__)


class OptimizerName(StrEnum):
    """Supported optimizer name strings."""

    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class BaseModel(ABC):
    """Abstract base class for all models.

    This class defines the interface that all models must implement,
    making it easy to create new model architectures.
    """

    def __init__(self, config: BaseConfig):
        """Initialize the model.

        Args:
            config: Configuration object
        """
        self.config = config
        self.model: tf.keras.Model | None = None

    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Build and return the model architecture.

        Returns:
            Compiled Keras model
        """
        pass

    def compile(
        self,
        optimizer: str | tf.keras.optimizers.Optimizer | None = None,
        loss: str | None = None,
        metrics: list[str] | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Compile the model.

        Args:
            optimizer: Keras optimizer (default: Adam with config learning rate)
            loss: Loss function (default: categorical_crossentropy)
            metrics: List of metrics (default: ['acc'])
            learning_rate: Optional override for optimizer learning rate
        """
        if self.model is None:
            self.model = self.build()

        lr = learning_rate if learning_rate is not None else self.config.learning_rate

        if optimizer is None:
            optimizer_obj = tf.keras.optimizers.Adam(learning_rate=lr)
        elif isinstance(optimizer, str):
            opt_name = optimizer.lower()
            if opt_name == OptimizerName.ADAM:
                optimizer_obj = tf.keras.optimizers.Adam(learning_rate=lr)
            elif opt_name == OptimizerName.SGD:
                optimizer_obj = tf.keras.optimizers.SGD(learning_rate=lr)
            elif opt_name == OptimizerName.RMSPROP:
                optimizer_obj = tf.keras.optimizers.RMSprop(learning_rate=lr)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        else:
            optimizer_obj = optimizer

        if loss is None:
            label_smoothing = getattr(self.config, "label_smoothing", 0.0) or 0.0
            if label_smoothing > 0:
                loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
            else:
                loss = "categorical_crossentropy"

        if metrics is None:
            metrics = ["accuracy"]

        if self.model is None:
            raise RuntimeError("Model must be built before compile")
        self.model.compile(optimizer=optimizer_obj, loss=loss, metrics=metrics)

    def summary(self) -> str:
        """Print and return model summary as a string.

        Returns:
            str: The textual summary of the model.
        """
        if self.model is None:
            self.model = self.build()
        if self.model is None:
            raise RuntimeError("Model build should return a valid model")

        lines = []
        self.model.summary(print_fn=lines.append)
        summary_text = "\n".join(lines)
        logger.info(summary_text)
        return summary_text

    def save(self, filepath: Path) -> None:
        """Save the model.

        Args:
            filepath: Path where model should be saved
        """
        if self.model is None:
            raise ValueError("Model not built yet")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info("Model saved to %s", filepath)

    def load(self, filepath: Path) -> tf.keras.Model:
        """Load a saved model and attach it to this instance.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded Keras model
        """
        loaded_model = tf.keras.models.load_model(filepath)
        self.model = loaded_model
        return loaded_model
