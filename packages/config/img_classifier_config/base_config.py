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
    working_dir: Path = Field(default=Path.home() / ".local" / "share" / "img_classifier")

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

    # Early-stopping tuning (defaults preserve the historical val_loss/min behaviour)
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"
    restore_best_weights: bool = False

    # Learning-rate scheduling (off by default)
    use_reduce_lr: bool = False
    reduce_lr_factor: float = Field(default=0.5, gt=0, lt=1)
    reduce_lr_patience: int = Field(default=4, ge=1)
    min_lr: float = Field(default=1e-6, ge=0)

    # Stop once validation accuracy reaches this target (None disables)
    target_val_accuracy: float | None = Field(default=None, ge=0, le=1)

    # Generalization knobs (all opt-in; defaults keep prior behaviour)
    use_data_augmentation: bool = False
    aug_rotation: float = Field(default=0.06, ge=0)
    aug_translation: float = Field(default=0.08, ge=0)
    aug_zoom: float = Field(default=0.10, ge=0)
    aug_horizontal_flip: bool = True
    use_class_weights: bool = False
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)

    # Force a Global Average Pooling head on the from-scratch CNN (instead of Flatten). GAP
    # collapses the spatial dims before the dense head, so the parameter count no longer explodes
    # with image size — from-scratch can train at 224px without the huge Flatten->Dense head OOM.
    # Default False preserves the per-complexity head choice (Simple/Medium=Flatten, Deep=GAP).
    use_global_pooling_head: bool = False

    # Transfer learning. backbone: None (from-scratch) | "mobilenetv2" | "efficientnetb0".
    backbone: str | None = None
    finetune_epochs: int = Field(default=0, ge=0)
    finetune_lr_divisor: float = Field(default=100.0, gt=0)

    # Evaluation-split determinism. A fixed seed makes the val/test partition identical across
    # runs/attempts (so model selection is apples-to-apples); stratification keeps class
    # balance in both halves. seed=None preserves the legacy unseeded random split.
    eval_split_seed: int | None = None
    stratify_eval_split: bool = False

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
