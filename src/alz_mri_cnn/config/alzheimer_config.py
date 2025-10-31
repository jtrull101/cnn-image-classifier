"""Configuration specific to the Alzheimer's MRI classification task."""

from dataclasses import dataclass
from pathlib import Path
from .base_config import BaseConfig


@dataclass
class AlzheimerConfig(BaseConfig):
    """Configuration for Alzheimer's MRI classification."""

    project_name: str = "alzheimer_mri_cnn"
    working_dir: Path = Path("/tmp/alz_mri_cnn/")

    dataset_name: str = "Combined Dataset"
    dataset_zip_id: str = "1SQuB_8IL3s7vZPMeGkOZo116QSTMa6BN"
    pretrained_model_id: str = "1U9uywbNatIFAj6XlahT6BBrMqyLgd4qZ"

    # Alzheimer's specific classes
    num_classes: int = 4
    class_names: list = None
    nice_class_names: list = None

    def __post_init__(self):
        """Initialize Alzheimer's specific settings."""
        super().__post_init__()

        # Set class names if not provided
        if self.class_names is None:
            self.class_names = [
                "MildDemented",
                "NonDemented",
                "ModerateDemented",
                "VeryMildDemented",
            ]

        if self.nice_class_names is None:
            self.nice_class_names = [
                "Mild Impairment",
                "No Impairment",
                "Moderate Impairment",
                "Very Mild Impairment",
            ]

