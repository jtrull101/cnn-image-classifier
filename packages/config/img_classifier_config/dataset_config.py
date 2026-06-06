"""Dataset configuration and auto-detection system."""

import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import Field, field_validator

from img_classifier_config.base_config import BaseConfig


logger = logging.getLogger(__name__)


class ArchitectureComplexity(StrEnum):
    """Valid values for the architecture_complexity field."""

    AUTO = "auto"
    SIMPLE = "simple"
    MEDIUM = "medium"
    DEEP = "deep"
    CUSTOM = "custom"


class DatasetConfig(BaseConfig):
    """Generic dataset configuration.

    This class provides dataset-agnostic configuration that can be used
    for any image classification task. It extends BaseConfig with additional
    fields for architecture hints and dataset characteristics.
    """

    # Dataset metadata
    dataset_type: str = Field(default="image_classification", description="Type of dataset")
    description: str | None = Field(default=None, description="Dataset description")

    # Architecture hints
    architecture_complexity: ArchitectureComplexity = Field(
        default=ArchitectureComplexity.AUTO,
        description="Model complexity: 'simple', 'medium', 'deep', or 'auto'",
    )
    recommended_batch_sizes: list[int] = Field(
        default_factory=lambda: [16, 32, 64], description="Recommended batch sizes for optimization"
    )
    recommended_learning_rates: list[float] = Field(
        default_factory=lambda: [0.0001, 0.001, 0.01],
        description="Recommended learning rates for optimization",
    )

    # Auto-detection settings
    auto_detect_classes: bool = Field(
        default=True, description="Automatically detect classes from directory structure"
    )
    min_images_per_class: int = Field(
        default=10, description="Minimum number of images required per class"
    )

    def model_post_init(self, __context) -> None:
        """Post-init hook to manage path defaults."""
        fields_set = getattr(self, "__pydantic_fields_set__", set())
        explicit_none_data = "data_path" in fields_set and self.data_path is None
        explicit_none_train = "train_path" in fields_set and self.train_path is None
        explicit_none_test = "test_path" in fields_set and self.test_path is None

        super().model_post_init(__context)

        # If caller explicitly set a path to None, preserve that instead of deriving defaults
        if explicit_none_data:
            self.data_path = None
            if not explicit_none_train:
                self.train_path = None
            if not explicit_none_test:
                self.test_path = None

        if explicit_none_train:
            self.train_path = None

        if explicit_none_test:
            self.test_path = None

    @field_validator("architecture_complexity", mode="before")
    @classmethod
    def validate_complexity(cls, v: str) -> str:
        """Validate and normalise architecture complexity value."""
        allowed = [e.value for e in ArchitectureComplexity]
        normalized = str(v).lower()
        if normalized not in allowed:
            raise ValueError(f"architecture_complexity must be one of {allowed}")
        return normalized

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DatasetConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            DatasetConfig instance
        """
        with yaml_path.open() as f:
            config_dict = yaml.safe_load(f)

        # Convert string paths back to Path objects
        path_fields = ["working_dir", "data_path", "train_path", "test_path"]
        for field in path_fields:
            if field in config_dict and config_dict[field] is not None:
                config_dict[field] = Path(config_dict[field])

        return cls(**config_dict)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path where YAML file should be saved
        """
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with yaml_path.open("w") as f:
            # Convert to dict and keep None values so explicit nulls roundtrip
            config_dict = self.model_dump(exclude_none=False, mode="json")

            # Convert Path objects to strings for YAML serialization
            def convert_paths(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                if isinstance(obj, Path):
                    return str(obj)
                return obj

            config_dict = convert_paths(config_dict)
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


class DatasetDetector:
    """Automatically detect dataset characteristics from filesystem.

    This class analyzes a dataset directory structure and extracts
    information about classes, image counts, and recommended settings.
    """

    SUPPORTED_IMAGE_EXTENSIONS: ClassVar[set[str]] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tiff",
    }

    def __init__(self, dataset_path: Path):
        """Initialize the dataset detector.

        Args:
            dataset_path: Root path of the dataset
        """
        self.dataset_path = Path(dataset_path)
        self._cached_info: dict[str, Any] | None = None

    def detect(self) -> dict[str, Any]:
        """Detect dataset characteristics (result is cached after the first call).

        Returns:
            Dictionary with dataset information
        """
        if self._cached_info is not None:
            return self._cached_info

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

        # Detect train/test structure
        has_train_test = self._detect_train_test_split()

        if has_train_test:
            train_path = self.dataset_path / "train"
            test_path = self.dataset_path / "test"

            train_classes, train_counts = self._detect_classes(train_path)
            test_classes, test_counts = self._detect_classes(test_path)

            # Ensure classes match
            if set(train_classes) != set(test_classes):
                logger.warning("Train and test classes don't match")
                classes = sorted(set(train_classes) | set(test_classes))
            else:
                classes = train_classes

            total_images = sum(train_counts.values()) + sum(test_counts.values())
            class_distribution = {
                cls: train_counts.get(cls, 0) + test_counts.get(cls, 0) for cls in classes
            }
        else:
            # No train/test split, analyze root directory
            classes, class_counts = self._detect_classes(self.dataset_path)
            total_images = sum(class_counts.values())
            class_distribution = class_counts

        # Detect image properties from sample
        sample_image_shape = self._detect_image_properties()

        # Calculate recommended settings
        complexity = self._recommend_complexity(total_images, len(classes))

        self._cached_info = {
            "dataset_path": str(self.dataset_path),
            "has_train_test_split": has_train_test,
            "num_classes": len(classes),
            "class_names": classes,
            "total_images": total_images,
            "class_distribution": class_distribution,
            "sample_image_shape": sample_image_shape,
            "recommended_complexity": complexity,
            "is_balanced": self._check_balance(class_distribution),
        }
        return self._cached_info

    def _detect_train_test_split(self) -> bool:
        """Check if dataset has train/test directory structure.

        Returns:
            True if train and test directories exist
        """
        train_path = self.dataset_path / "train"
        test_path = self.dataset_path / "test"
        return train_path.exists() and test_path.exists()

    def _detect_classes(self, path: Path) -> tuple[list[str], dict[str, int]]:
        """Detect classes from directory structure.

        Args:
            path: Directory to analyze

        Returns:
            Tuple of (class_names, class_counts)
        """
        if not path.exists():
            return [], {}

        classes = []
        class_counts = {}

        for item in sorted(path.iterdir()):
            if item.is_dir():
                # Count images in this class directory
                image_count = sum(
                    1 for f in item.iterdir() if f.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS
                )

                if image_count > 0:
                    classes.append(item.name)
                    class_counts[item.name] = image_count

        return classes, class_counts

    def _detect_image_properties(self) -> tuple[int, int, int] | None:
        """Detect image properties from a sample image.

        Returns:
            Tuple of (height, width, channels) or None if no images found
        """
        try:
            import cv2

            # Find first image in dataset
            for root, _, files in os.walk(self.dataset_path):
                for file in files:
                    if Path(file).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                        img_path = Path(root) / file
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            return tuple(img.shape)
            return None
        except Exception as e:
            logger.warning("Could not detect image properties: %s", e)
            return None

    def _recommend_complexity(self, total_images: int, num_classes: int) -> ArchitectureComplexity:
        """Recommend model complexity based on dataset size.

        Args:
            total_images: Total number of images
            num_classes: Number of classes

        Returns:
            Recommended complexity enum value
        """
        if total_images < 1000 or num_classes <= 3:
            return ArchitectureComplexity.SIMPLE
        if total_images < 10000 or num_classes <= 10:
            return ArchitectureComplexity.MEDIUM
        return ArchitectureComplexity.DEEP

    def _check_balance(self, class_distribution: dict[str, int]) -> bool:
        """Check if dataset is balanced.

        Args:
            class_distribution: Dictionary of class counts

        Returns:
            True if dataset is reasonably balanced
        """
        if not class_distribution:
            return True

        counts = list(class_distribution.values())
        min_count = min(counts)
        max_count = max(counts)

        # Consider balanced if the ratio is within 2:1
        return max_count / min_count <= 2.0

    def create_config(
        self, project_name: str | None = None, working_dir: Path | None = None, **kwargs
    ) -> DatasetConfig:
        """Create a DatasetConfig from detected characteristics.

        Args:
            project_name: Name for the project
            working_dir: Working directory for outputs
            **kwargs: Additional config parameters to override

        Returns:
            DatasetConfig instance
        """
        info = self.detect()

        config_dict: dict[str, Any] = {
            "project_name": project_name or self.dataset_path.name,
            "working_dir": working_dir
            or Path.home() / ".local" / "share" / "img_classifier" / self.dataset_path.name,
            "dataset_name": self.dataset_path.name,
            "data_path": self.dataset_path,
            "num_classes": info["num_classes"],
            "class_names": info["class_names"],
            "architecture_complexity": info["recommended_complexity"],
        }

        # Add a sample image shape if detected
        if info["sample_image_shape"]:
            h, w, c = info["sample_image_shape"]
            # Use detected size or standardize to a reasonable size
            if h > 256 or w > 256:
                config_dict["image_size"] = (128, 128)
            else:
                config_dict["image_size"] = (h, w)
            config_dict["color_channels"] = c

        # Override with any user-provided kwargs
        config_dict.update(kwargs)

        return DatasetConfig(**config_dict)
