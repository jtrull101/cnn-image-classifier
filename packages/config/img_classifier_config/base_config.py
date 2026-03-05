"""Base configuration class for all models using Pydantic."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """Base configuration for model training and inference.

    Uses Pydantic for validation and settings management.
    Can be extended to create custom configurations for different datasets.
    Supports loading from environment variables with IMG_CLASSIFIER_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="IMG_CLASSIFIER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # Project settings
    project_name: str = "img_classifier"
    working_dir: Path = Field(default=Path("/tmp/img_classifier/"))

    # Data settings
    dataset_name: str = "dataset"
    dataset_zip_id: str | None = None  # Google Drive ID
    data_path: Path | None = None
    train_path: Path | None = None
    test_path: Path | None = None

    # Image settings
    image_size: tuple[int, int] = (128, 128)
    color_channels: int = Field(default=3, ge=1, le=4)

    # Class settings
    num_classes: int = Field(default=4, ge=0)
    class_names: list[str] = Field(default_factory=list)

    # Training hyperparameters
    batch_size: int = Field(default=32, ge=1)
    num_epochs: int = Field(default=25, ge=1)
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    validation_split: float = Field(default=0.2, ge=0, lt=1)
    test_split: float = Field(default=0.5, ge=0, lt=1)

    # Data sampling
    data_percent: float = Field(default=1.0, gt=0, le=1)

    # Model settings
    dropout_rate: float = Field(default=0.3, ge=0, lt=1)
    early_stopping_patience: int = Field(default=20, ge=1)
    min_accuracy_to_save: float = Field(default=0.98, ge=0, le=1)

    # Callbacks
    use_early_stopping: bool = True
    use_checkpointing: bool = True

    def model_post_init(self, __context) -> None:
        """Initialize paths after model creation."""
        # Create derived paths if not set
        if self.data_path is None:
            self.data_path = self.working_dir / "data" / self.dataset_name

        if self.train_path is None:
            self.train_path = self.data_path / "train"

        if self.test_path is None:
            self.test_path = self.data_path / "test"

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Get the input shape for the model."""
        return (*self.image_size, self.color_channels)

    @property
    def models_dir(self) -> Path:
        """Get the models directory path."""
        return self.working_dir / "models"

    @property
    def logs_dir(self) -> Path:
        """Get the logs directory path."""
        return self.working_dir / "logs"

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory for pickled data.

        Requires data_path to be set (typically via DatasetConfig.model_post_init).
        Raises RuntimeError if called before data_path is initialised.
        """
        if self.data_path is None:
            raise RuntimeError(
                "cache_dir is unavailable: data_path has not been set. "
                "Ensure model_post_init has been called or pass dataset_path to the config."
            )
        return self.data_path / "cache"

    def create_directories(self) -> None:
        """Create all necessary directories."""
        for directory in [
            self.working_dir,
            self.models_dir,
            self.logs_dir,
            self.cache_dir,
            self.data_path,
        ]:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)

    use_model_checkpoint: bool = True
    use_accuracy_threshold_stopping: bool = True
    accuracy_threshold: float = 0.995
