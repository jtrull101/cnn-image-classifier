"""Dynamic CNN architecture generator based on dataset characteristics."""

import math
from enum import Enum
from typing import Dict, List, Optional

import tensorflow as tf
from img_classifier_config import BaseConfig
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from keras.models import Sequential


class ArchitectureComplexity(Enum):
    """Enum for architecture complexity levels."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    DEEP = "deep"
    CUSTOM = "custom"


class ArchitectureSpec:
    """Specification for CNN architecture.

    This class defines the structure of a CNN including the number of
    convolutional blocks, filter counts, and dense layer sizes.
    """

    def __init__(
        self,
        name: str,
        conv_blocks: List[Dict],
        dense_layers: List[int],
        use_batch_norm: bool = False,
        use_global_pooling: bool = False,
        dropout_rate: float = 0.3,
    ):
        """Initialize architecture specification.

        Args:
            name: Name of the architecture
            conv_blocks: List of dicts with conv block specs
                Each dict should have: filters, kernel_size, activation, pooling
            dense_layers: List of dense layer sizes (excluding output layer)
            use_batch_norm: Whether to use batch normalization
            use_global_pooling: Whether to use global average pooling instead of flatten
            dropout_rate: Dropout rate to apply
        """
        self.name = name
        self.conv_blocks = conv_blocks
        self.dense_layers = dense_layers
        self.use_batch_norm = use_batch_norm
        self.use_global_pooling = use_global_pooling
        self.dropout_rate = dropout_rate


class ArchitectureFactory:
    """Factory for generating CNN architectures based on dataset characteristics."""

    @staticmethod
    def create(
        config: BaseConfig,
        complexity: Optional[str] = None,
        custom_spec: Optional[ArchitectureSpec] = None,
    ) -> tf.keras.Model:
        """Create a CNN model based on configuration and complexity.

        Args:
            config: Configuration object with dataset info
            complexity: Complexity level ('simple', 'medium', 'deep', 'auto')
                If None, uses config.architecture_complexity if available
            custom_spec: Custom architecture specification

        Returns:
            Compiled Keras model
        """
        if custom_spec is not None:
            return ArchitectureFactory._build_from_spec(config, custom_spec)

        # Determine complexity
        if complexity is None:
            complexity = str(getattr(config, "architecture_complexity", "auto"))

        if complexity == "auto":
            complexity = ArchitectureFactory._auto_select_complexity(config)

        # Get architecture spec based on complexity
        spec = ArchitectureFactory._get_spec_for_complexity(
            complexity, config.input_shape, config.num_classes
        )

        return ArchitectureFactory._build_from_spec(config, spec)

    @staticmethod
    def _auto_select_complexity(config: BaseConfig) -> str:
        """Automatically select complexity based on dataset characteristics.

        Args:
            config: Configuration object

        Returns:
            Recommended complexity level
        """
        # Use image size and number of classes as heuristics
        image_area = config.image_size[0] * config.image_size[1]

        if config.num_classes <= 3 and image_area <= 64 * 64:
            return "simple"
        elif config.num_classes <= 10 and image_area <= 128 * 128:
            return "medium"
        else:
            return "deep"

    @staticmethod
    def _get_spec_for_complexity(
        complexity: str,
        input_shape: tuple,
        num_classes: int,
    ) -> ArchitectureSpec:
        """Get architecture specification for given complexity.

        Args:
            complexity: Complexity level
            input_shape: Input shape tuple
            num_classes: Number of output classes

        Returns:
            ArchitectureSpec instance
        """
        if complexity == "simple":
            return ArchitectureSpec(
                name="SimpleCNN",
                conv_blocks=[
                    {"filters": 32, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
                    {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
                    {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
                ],
                dense_layers=[64],
                use_batch_norm=False,
                use_global_pooling=False,
                dropout_rate=0.3,
            )

        elif complexity == "medium":
            return ArchitectureSpec(
                name="MediumCNN",
                conv_blocks=[
                    {"filters": 64, "kernel_size": (5, 5), "activation": "relu", "pooling": True},
                    {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
                    {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
                    {"filters": 128, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
                ],
                dense_layers=[128],
                use_batch_norm=False,
                use_global_pooling=False,
                dropout_rate=0.3,
            )

        elif complexity == "deep":
            return ArchitectureSpec(
                name="DeepCNN",
                conv_blocks=[
                    {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
                    {"filters": 128, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
                    {"filters": 128, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
                    {"filters": 256, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
                    {"filters": 256, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
                ],
                dense_layers=[256, 128],
                use_batch_norm=True,
                use_global_pooling=True,
                dropout_rate=0.4,
            )

        else:
            raise ValueError(f"Unknown complexity: {complexity}")

    @staticmethod
    def _build_from_spec(config: BaseConfig, spec: ArchitectureSpec) -> tf.keras.Model:
        """Build model from architecture specification.

        Args:
            config: Configuration object
            spec: Architecture specification

        Returns:
            Keras Sequential model
        """
        layers = [Input(shape=config.input_shape)]

        # Add convolutional blocks
        for i, block in enumerate(spec.conv_blocks):
            layers.append(
                Conv2D(
                    filters=block["filters"],
                    kernel_size=block["kernel_size"],
                    activation=block["activation"],
                    padding="same",
                )
            )

            # Add batch normalization if enabled
            if spec.use_batch_norm:
                layers.append(BatchNormalization())

            # Add pooling
            if block.get("pooling", True):
                layers.append(MaxPooling2D())

            # Add dropout for deeper layers
            if i >= len(spec.conv_blocks) - 2 and spec.dropout_rate > 0:
                layers.append(Dropout(spec.dropout_rate))

        # Add flattening or global pooling
        if spec.use_global_pooling:
            layers.append(GlobalAveragePooling2D())
        else:
            layers.append(Flatten())

        # Add dense layers
        for dense_size in spec.dense_layers:
            layers.append(Dense(dense_size, activation="relu"))
            if spec.dropout_rate > 0:
                layers.append(Dropout(spec.dropout_rate))

        # Add output layer
        layers.append(Dense(config.num_classes, activation="softmax"))

        model = Sequential(layers, name=spec.name)
        return model


class ModelScaler:
    """Utility for scaling model architectures based on dataset size."""

    @staticmethod
    def scale_filters(
        base_filters: int,
        dataset_size: int,
        scale_factor: float = 1.0,
    ) -> int:
        """Scale number of filters based on dataset size.

        Args:
            base_filters: Base number of filters
            dataset_size: Size of dataset (number of samples)
            scale_factor: Additional scaling factor

        Returns:
            Scaled filter count
        """
        if dataset_size < 1000:
            multiplier = 0.5
        elif dataset_size < 5000:
            multiplier = 0.75
        elif dataset_size < 20000:
            multiplier = 1.0
        else:
            multiplier = 1.5

        scaled = int(base_filters * multiplier * scale_factor)
        # Round to nearest power of 2
        log_val = math.log2(float(scaled))
        return 2 ** round(log_val)

    @staticmethod
    def recommend_depth(
        image_size: tuple,
        num_classes: int,
        dataset_size: int,
    ) -> int:
        """Recommend number of convolutional blocks.

        Args:
            image_size: Size of input images (height, width)
            num_classes: Number of output classes
            dataset_size: Size of dataset

        Returns:
            Recommended number of conv blocks
        """
        max_dimension = max(image_size)

        # Calculate maximum useful depth based on image size
        # Each pooling layer reduces dimension by 2
        max_depth = int(math.log2(float(max_dimension))) - 2

        # Adjust based on dataset size and complexity
        if dataset_size < 1000:
            recommended = min(3, max_depth)
        elif dataset_size < 5000:
            recommended = min(4, max_depth)
        else:
            recommended = min(5, max_depth)

        # Adjust for number of classes
        if num_classes > 20:
            recommended = min(recommended + 1, max_depth)

        return max(2, recommended)  # At least 2 blocks


def create_architecture_from_config(config: BaseConfig) -> tf.keras.Model:
    """Convenience function to create architecture from config.

    Args:
        config: Configuration object

    Returns:
        Keras model
    """
    return ArchitectureFactory.create(config)
