"""Base configuration class for all models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional


@dataclass
class BaseConfig:
    """Base configuration for model training and inference.

    This class can be extended to create custom configurations for different
    datasets and model architectures.
    """

    # Project settings
    project_name: str = "mri_classifier"
    working_dir: Path = field(default_factory=lambda: Path("/tmp/mri_classifier/"))

    # Data settings
    dataset_name: str = "dataset"
    dataset_zip_id: Optional[str] = None  # Google Drive ID
    data_path: Optional[Path] = None
    train_path: Optional[Path] = None
    test_path: Optional[Path] = None

    # Image settings
    image_size: Tuple[int, int] = (128, 128)
    color_channels: int = 3

    # Class settings
    num_classes: int = 4
    class_names: List[str] = field(default_factory=list)

    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 25
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_split: float = 0.5  # Split of test data for validation

    # Data sampling
    data_percent: float = 1.0  # Percentage of dataset to use

    # Model settings
    dropout_rate: float = 0.3
    early_stopping_patience: int = 20
    min_accuracy_to_save: float = 0.98

    # Callbacks
    use_early_stopping: bool = True
    use_model_checkpoint: bool = True
    use_accuracy_threshold_stopping: bool = True
    accuracy_threshold: float = 0.995

    def __post_init__(self):
        """Initialize derived paths after instance creation."""
        # Convert strings to Path objects
        if isinstance(self.working_dir, str):
            self.working_dir = Path(self.working_dir)

        # Set default paths if not provided
        if self.data_path is None:
            self.data_path = self.working_dir / "data"
        if self.train_path is None:
            self.train_path = self.data_path / "train"
        if self.test_path is None:
            self.test_path = self.data_path / "test"

        # Ensure paths are Path objects
        self.data_path = Path(self.data_path)
        self.train_path = Path(self.train_path)
        self.test_path = Path(self.test_path)

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Get the input shape for the model."""
        return (*self.image_size, self.color_channels)

    @property
    def models_dir(self) -> Path:
        """Get the models directory."""
        return self.working_dir / "models"

    @property
    def logs_dir(self) -> Path:
        """Get the logs directory."""
        return self.working_dir / "logs"

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory for pickled data."""
        return self.data_path / "cache"

    def create_directories(self):
        """Create all required directories."""
        for directory in [
            self.working_dir,
            self.data_path,
            self.models_dir,
            self.logs_dir,
            self.cache_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

