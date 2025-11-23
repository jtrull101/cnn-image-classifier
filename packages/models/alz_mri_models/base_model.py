"""Base model interface."""

from abc import ABC, abstractmethod
from typing import Optional
import tensorflow as tf
from pathlib import Path

from alz_mri_config import BaseConfig


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
        self.model: Optional[tf.keras.Model] = None

    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Build and return the model architecture.

        Returns:
            Compiled Keras model
        """
        pass

    def compile(
        self,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss: Optional[str] = None,
        metrics: Optional[list] = None,
    ):
        """Compile the model.

        Args:
            optimizer: Keras optimizer (default: Adam with config learning rate)
            loss: Loss function (default: categorical_crossentropy)
            metrics: List of metrics (default: ['acc'])
        """
        if self.model is None:
            self.model = self.build()

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        if loss is None:
            loss = "categorical_crossentropy"

        if metrics is None:
            metrics = ["acc"]

        assert self.model is not None, "Model must be built before compile"
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.model = self.build()
        assert self.model is not None, "Model build should return a valid model"
        return self.model.summary()

    def save(self, filepath: Path):
        """Save the model.

        Args:
            filepath: Path where model should be saved
        """
        if self.model is None:
            raise ValueError("Model not built yet")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> tf.keras.Model:
        """Load a saved model.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded Keras model
        """
        return tf.keras.models.load_model(filepath)
