"""Base model interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import tensorflow as tf
from img_classifier_config import BaseConfig


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
        optimizer: Optional[Union[str, tf.keras.optimizers.Optimizer]] = None,
        loss: Optional[str] = None,
        metrics: Optional[list] = None,
        learning_rate: Optional[float] = None,
    ):
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
            if opt_name == "adam":
                optimizer_obj = tf.keras.optimizers.Adam(learning_rate=lr)
            elif opt_name == "sgd":
                optimizer_obj = tf.keras.optimizers.SGD(learning_rate=lr)
            elif opt_name == "rmsprop":
                optimizer_obj = tf.keras.optimizers.RMSprop(learning_rate=lr)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        else:
            optimizer_obj = optimizer

        if loss is None:
            loss = "categorical_crossentropy"

        if metrics is None:
            metrics = ["accuracy"]

        if self.model is None:
            raise RuntimeError("Model must be built before compile")
        self.model.compile(optimizer=optimizer_obj, loss=loss, metrics=metrics)

    def summary(self):
        """Print and return model summary as a string.

        Returns:
            str: The textual summary of the model.
        """
        if self.model is None:
            self.model = self.build()
        if self.model is None:
            raise RuntimeError("Model build should return a valid model")

        # Capture Keras summary into a string and also print it to stdout
        lines = []
        self.model.summary(print_fn=lines.append)
        summary_text = "\n".join(lines)
        # Ensure existing behavior of printing to stdout for tests capturing output
        print(summary_text)
        return summary_text

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
