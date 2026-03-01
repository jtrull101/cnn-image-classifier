"""Configuration specific to the Alzheimer's MRI classification task using Pydantic."""

from pathlib import Path
from typing import List

from pydantic import Field

from .base_config import BaseConfig

# TODO: can this be removed?
class AlzheimerConfig(BaseConfig):
    """Configuration for Alzheimer's MRI classification.

    Extends BaseConfig with Alzheimer's specific settings.
    """

    project_name: str = "alzheimer_mri_cnn"
    working_dir: Path = Path("/tmp/img_classifier_cnn/")

    dataset_name: str = "Combined Dataset"
    dataset_zip_id: str = "1SQuB_8IL3s7vZPMeGkOZo116QSTMa6BN"
    pretrained_model_id: str = "1U9uywbNatIFAj6XlahT6BBrMqyLgd4qZ"

    # Alzheimer's specific classes
    num_classes: int = 4
    class_names: List[str] = Field(
        default_factory=lambda: [
            "MildDemented",
            "NonDemented",
            "ModerateDemented",
            "VeryMildDemented",
        ]
    )
    nice_class_names: List[str] = Field(
        default_factory=lambda: [
            "Mild Impairment",
            "No Impairment",
            "Moderate Impairment",
            "Very Mild Impairment",
        ]
    )
