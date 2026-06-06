"""Dynamic CNN architecture generator based on dataset characteristics."""

import math
from typing import ClassVar

import tensorflow as tf
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

from img_classifier_config import ArchitectureComplexity, BaseConfig


# Thresholds for automatic complexity selection (used by _auto_select_complexity)
_SIMPLE_MAX_CLASSES = 3
_MEDIUM_MAX_CLASSES = 10
_SIMPLE_MAX_IMAGE_AREA = 64 * 64  # 4 096 px²
_MEDIUM_MAX_IMAGE_AREA = 128 * 128  # 16 384 px²

# Thresholds for filter scaling and depth recommendation (ModelScaler)
_SCALE_SMALL_DATASET = 1_000
_SCALE_MEDIUM_DATASET = 5_000
_SCALE_LARGE_DATASET = 20_000
_DEPTH_SMALL_DATASET = 1_000
_DEPTH_MEDIUM_DATASET = 5_000
_DEPTH_MANY_CLASSES = 20


class ArchitectureSpec:
    """Specification for CNN architecture.

    This class defines the structure of a CNN including the number of
    convolutional blocks, filter counts, and dense layer sizes.
    """

    def __init__(
        self,
        name: str,
        conv_blocks: list[dict],
        dense_layers: list[int],
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
        complexity: ArchitectureComplexity | None = None,
        custom_spec: ArchitectureSpec | None = None,
    ) -> tf.keras.Model:
        """Create a CNN model based on configuration and complexity.

        Args:
            config: Configuration object with dataset info
            complexity: Complexity level. If None, uses config.architecture_complexity if available.
            custom_spec: Custom architecture specification

        Returns:
            Compiled Keras model
        """
        if custom_spec is not None:
            return ArchitectureFactory._build_from_spec(config, custom_spec)

        # Transfer learning: a pretrained backbone takes precedence over the from-scratch CNN.
        if getattr(config, "backbone", None):
            model, _base = ArchitectureFactory.create_backbone(config)
            return model

        # Determine complexity
        if complexity is None:
            complexity = ArchitectureComplexity(
                getattr(config, "architecture_complexity", ArchitectureComplexity.AUTO)
            )

        if complexity == ArchitectureComplexity.AUTO:
            complexity = ArchitectureFactory._auto_select_complexity(config)

        # Get architecture spec based on complexity
        spec = ArchitectureFactory._get_spec_for_complexity(
            complexity, config.input_shape, config.num_classes
        )

        # Opt-in: force a GAP head so the from-scratch dense head doesn't explode with image size
        # (lets from-scratch train at 224px without the Flatten->Dense OOM).
        if getattr(config, "use_global_pooling_head", False):
            spec.use_global_pooling = True

        return ArchitectureFactory._build_from_spec(config, spec)

    # Backbones whose ImageNet weights expect raw [0, 255] inputs (preprocessing is internal);
    # everything else here is fed via an explicit Rescaling layer below.
    _BACKBONES: ClassVar[dict[str, tuple[str, tuple[float, float]]]] = {
        "mobilenetv2": ("MobileNetV2", (2.0, -1.0)),  # scale [0,1] -> [-1,1]
        "efficientnetb0": ("EfficientNetB0", (255.0, 0.0)),  # scale [0,1] -> [0,255]
    }

    @staticmethod
    def create_backbone(config: BaseConfig) -> tuple[tf.keras.Model, tf.keras.layers.Layer]:
        """Build a transfer-learning model around a pretrained ImageNet backbone.

        The app feeds models pixels in [0, 1] (DJL Resize -> ToTensor), so a ``Rescaling``
        layer inside the graph maps that into each backbone's expected input domain. The base
        is returned alongside the model so the caller can unfreeze it for fine-tuning.

        Args:
            config: Configuration with ``backbone``, ``input_shape``, ``num_classes``.

        Returns:
            Tuple of (model, base_layer).
        """
        from keras import applications
        from keras.layers import Dense, Dropout, Input, Rescaling

        key = str(config.backbone).lower()
        if key not in ArchitectureFactory._BACKBONES:
            raise ValueError(
                f"Unknown backbone '{config.backbone}'. "
                f"Supported: {sorted(ArchitectureFactory._BACKBONES)}"
            )
        app_name, (scale, offset) = ArchitectureFactory._BACKBONES[key]
        app_cls = getattr(applications, app_name)

        inputs = Input(shape=config.input_shape)
        x = Rescaling(scale, offset=offset)(inputs)
        base = app_cls(include_top=False, weights="imagenet", pooling="avg", input_tensor=x)
        base.trainable = False
        y = Dropout(config.dropout_rate)(base.output)
        outputs = Dense(config.num_classes, activation="softmax")(y)
        model = tf.keras.Model(inputs, outputs, name=f"{app_name}_transfer")
        return model, base

    @staticmethod
    def _auto_select_complexity(config: BaseConfig) -> ArchitectureComplexity:
        """Automatically select complexity based on dataset characteristics.

        Args:
            config: Configuration object

        Returns:
            Recommended complexity level
        """
        # Use image size and number of classes as heuristics
        image_area = config.image_size[0] * config.image_size[1]

        if config.num_classes <= _SIMPLE_MAX_CLASSES and image_area <= _SIMPLE_MAX_IMAGE_AREA:
            return ArchitectureComplexity.SIMPLE
        if config.num_classes <= _MEDIUM_MAX_CLASSES and image_area <= _MEDIUM_MAX_IMAGE_AREA:
            return ArchitectureComplexity.MEDIUM
        return ArchitectureComplexity.DEEP

    @staticmethod
    def _get_spec_for_complexity(
        complexity: ArchitectureComplexity,
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
        if complexity == ArchitectureComplexity.SIMPLE:
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

        if complexity == ArchitectureComplexity.MEDIUM:
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

        if complexity == ArchitectureComplexity.DEEP:
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

        return Sequential(layers, name=spec.name)


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
        if dataset_size < _SCALE_SMALL_DATASET:
            multiplier = 0.5
        elif dataset_size < _SCALE_MEDIUM_DATASET:
            multiplier = 0.75
        elif dataset_size < _SCALE_LARGE_DATASET:
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
        if dataset_size < _DEPTH_SMALL_DATASET:
            recommended = min(3, max_depth)
        elif dataset_size < _DEPTH_MEDIUM_DATASET:
            recommended = min(4, max_depth)
        else:
            recommended = min(5, max_depth)

        # Adjust for number of classes
        if num_classes > _DEPTH_MANY_CLASSES:
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
