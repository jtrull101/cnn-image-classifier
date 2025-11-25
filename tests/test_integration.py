"""Integration tests for the rearchitected modules.

These tests verify that all modules work together correctly, including:
- Module import validation (ensuring all public APIs are accessible)
- Config and data loader integration (verifying components interact properly)
- Overall package structure validation
"""

import pytest

from img_classifier_config import DatasetConfig


class TestArchitectureIntegration:
    """Tests for overall architecture integration."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures."""
        self.temp_dir = tmp_path
        yield


    def test_config_imports(self):
        """Test that all config classes can be imported."""
        from img_classifier_config import BaseConfig, DatasetConfig, DatasetDetector
        assert BaseConfig is not None
        assert DatasetConfig is not None
        assert DatasetDetector is not None

    def test_data_loader_imports(self):
        """Test that all data loader classes can be imported."""
        from img_classifier_data import BaseDataLoader, ImageDataLoader
        assert BaseDataLoader is not None
        assert ImageDataLoader is not None

    def test_utils_imports(self):
        """Test that all utility functions can be imported."""
        from img_classifier_utils import (
            clean_directory,
            download_from_google_drive,
            ensure_directory_exists,
            extract_archive,
            organize_dataset,
        )
        assert download_from_google_drive is not None
        assert extract_archive is not None
        assert organize_dataset is not None
        assert clean_directory is not None
        assert ensure_directory_exists is not None

    def test_config_and_data_loader_integration(self):
        """Test that config and data loader work together."""
        from img_classifier_config import BaseConfig
        from img_classifier_data import ImageDataLoader

        config = BaseConfig(working_dir=self.temp_dir)
        loader = ImageDataLoader(config)

        assert loader.config == config
        assert config.working_dir.exists()

    def test_dataset_config_initialization(self):
        """Test that DatasetConfig initializes with all required fields."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            num_classes=4,
            class_names=["class1", "class2", "class3", "class4"]
        )

        # Verify all expected fields are present
        assert config.num_classes == 4
        assert config.class_names is not None
        assert len(config.class_names) == 4

    def test_module_structure(self):
        """Test that the module structure is properly organized."""
        import img_classifier_config
        import img_classifier_data

        # Verify __all__ exports
        assert 'BaseConfig' in img_classifier_config.__all__
        assert 'DatasetConfig' in img_classifier_config.__all__
        assert 'BaseDataLoader' in img_classifier_data.__all__
        assert 'ImageDataLoader' in img_classifier_data.__all__

    def test_new_components_imports(self):
        """Test that new generalized components can be imported."""
        from img_classifier_models import ArchitectureFactory
        from img_classifier_training import HyperparameterOptimizer, TrainingOrchestrator

        assert ArchitectureFactory is not None
        assert TrainingOrchestrator is not None
        assert HyperparameterOptimizer is not None

