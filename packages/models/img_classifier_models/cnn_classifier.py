"""CNN classifier model for image classification."""

from typing import Optional

import tensorflow as tf
from img_classifier_config import ArchitectureComplexity, BaseConfig
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential

from .base_model import BaseModel


class CnnClassifier(BaseModel):
    """Convolutional Neural Network for image classification.

    This model uses a series of convolutional and pooling layers
    followed by fully connected layers for classification.
    """

    def __init__(self, config: BaseConfig, seed: int = 1234):
        """Initialize the CNN classifier.

        Args:
            config: Configuration object
            seed: Random seed for reproducibility
        """
        super().__init__(config)
        self.seed = seed
        tf.random.set_seed(seed)

    def build(
        self, complexity: Optional[ArchitectureComplexity] = None, seed: Optional[int] = None
    ) -> tf.keras.Model:
        """Build the CNN architecture.

        Args:
            complexity: Optional complexity hint ('simple', 'medium', 'deep')
            seed: Optional random seed override

        Returns:
            Keras Sequential model
        """
        if seed is not None:
            tf.random.set_seed(seed)
            self.seed = seed

        complexity_key = str(
            complexity
            or getattr(self.config, "architecture_complexity", ArchitectureComplexity.MEDIUM)
            or ArchitectureComplexity.MEDIUM
        )
        filters_by_complexity = {
            "simple": [32, 64, 64, 128],
            "medium": [64, 64, 64, 128],
            "deep": [64, 64, 128, 256],
        }
        filters = filters_by_complexity.get(complexity_key, filters_by_complexity["medium"])

        layers = [Input(shape=self.config.input_shape)]
        for filters_count in filters:
            layers.append(Conv2D(filters_count, (3, 3), activation="relu", padding="same"))
            layers.append(MaxPooling2D())
            if filters_count >= 128:
                layers.append(Dropout(self.config.dropout_rate))

        layers.extend(
            [
                Flatten(),
                Dense(self.config.image_size[0], activation="relu"),
                Dense(self.config.num_classes, activation="softmax"),
            ]
        )

        model = Sequential(layers)

        self.model = model
        return model


class SimpleCnn(BaseModel):
    """Simpler CNN for faster training and testing.

    This is a lighter version useful for quick experiments.
    """

    def __init__(self, config: BaseConfig, seed: int = 1234):
        """Initialize the simple CNN.

        Args:
            config: Configuration object
            seed: Random seed for reproducibility
        """
        super().__init__(config)
        self.seed = seed
        tf.random.set_seed(seed)

    def build(self) -> tf.keras.Model:
        """Build a simpler CNN architecture.

        Returns:
            Keras Sequential model
        """
        model = Sequential(
            [
                Input(shape=self.config.input_shape),
                # First conv block
                Conv2D(32, (3, 3), activation="relu"),
                MaxPooling2D(),
                # Second conv block
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D(),
                # Third conv block
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D(),
                # Flatten and dense layers
                Flatten(),
                Dense(64, activation="relu"),
                Dropout(self.config.dropout_rate),
                Dense(self.config.num_classes, activation="softmax"),
            ]
        )

        self.model = model
        return model
