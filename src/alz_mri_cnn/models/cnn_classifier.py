"""CNN classifier model for image classification."""

import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

from .base_model import BaseModel
from ..config import BaseConfig


class CNNClassifier(BaseModel):
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

    def build(self) -> tf.keras.Model:
        """Build the CNN architecture.

        Returns:
            Keras Sequential model
        """
        model = Sequential([
            # First conv block
            Conv2D(
                64, (5, 5),
                activation='relu',
                input_shape=self.config.input_shape
            ),
            MaxPooling2D(),

            # Second conv block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(),

            # Third conv block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(),

            # Fourth conv block with dropout
            Conv2D(128, (3, 3), activation='relu'),
            Dropout(self.config.dropout_rate),
            MaxPooling2D(),

            # Fifth conv block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(),

            # Flatten and dense layers
            Flatten(),
            Dense(self.config.image_size[0], activation='relu'),
            Dense(self.config.num_classes, activation='softmax'),
        ])

        self.model = model
        return model


class SimpleCNN(BaseModel):
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
        model = Sequential([
            # First conv block
            Conv2D(
                32, (3, 3),
                activation='relu',
                input_shape=self.config.input_shape
            ),
            MaxPooling2D(),

            # Second conv block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(),

            # Third conv block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(),

            # Flatten and dense layers
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(self.config.num_classes, activation='softmax'),
        ])

        self.model = model
        return model
