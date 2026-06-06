"""Integration tests for cross-package functionality.

These tests verify that multiple packages work together correctly, including:
- Module import validation (ensuring all public APIs are accessible)
- Config and data loader integration (verifying components interact properly)
- Overall package structure validation

These tests use real dependencies and may take longer to execute.
"""

import pytest

from img_classifier_config import BaseConfig, DatasetConfig, DatasetDetector
from img_classifier_data import BaseDataLoader, ImageDataLoader
from img_classifier_utils import (
    clean_directory,
    download_from_google_drive,
    ensure_directory_exists,
    extract_archive,
    organize_dataset,
)


pytestmark = pytest.mark.integration


class TestModuleImports:
    """Tests for verifying all modules can be imported correctly."""

    @pytest.mark.smoke
    def test_config_imports(self):
        """Test that all config classes can be imported."""
        assert BaseConfig is not None
        assert DatasetConfig is not None
        assert DatasetDetector is not None

    @pytest.mark.smoke
    def test_data_loader_imports(self):
        """Test that all data loader classes can be imported."""
        assert BaseDataLoader is not None
        assert ImageDataLoader is not None

    @pytest.mark.smoke
    def test_utils_imports(self):
        """Test that all utility functions can be imported."""
        assert download_from_google_drive is not None
        assert extract_archive is not None
        assert organize_dataset is not None
        assert clean_directory is not None
        assert ensure_directory_exists is not None

    @pytest.mark.smoke
    def test_models_imports(self):
        """Test that model components can be imported."""
        try:
            from img_classifier_models import ArchitectureFactory, BaseModel

            assert ArchitectureFactory is not None
            assert BaseModel is not None
        except ImportError:
            pytest.skip("TensorFlow not available")

    @pytest.mark.smoke
    def test_training_imports(self):
        """Test that training components can be imported."""
        try:
            from img_classifier_training import (
                HyperparameterOptimizer,
                Trainer,
                TrainingOrchestrator,
            )

            assert Trainer is not None
            assert TrainingOrchestrator is not None
            assert HyperparameterOptimizer is not None
        except ImportError:
            pytest.skip("TensorFlow not available")


class TestCrossPackageIntegration:
    """Tests for integration between multiple packages."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        return

    def test_config_and_data_loader_integration(self):
        """Test that config and data loader work together."""
        config = BaseConfig(working_dir=self.temp_dir)
        loader = ImageDataLoader(config)

        assert loader.config == config
        assert config.working_dir.exists()
        assert loader.config.data_path == config.data_path

    def test_dataset_config_initialization(self):
        """Test that DatasetConfig initializes with all required fields."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            num_classes=4,
            class_names=["class1", "class2", "class3", "class4"],
        )

        # Verify all expected fields are present
        assert config.num_classes == 4
        assert config.class_names is not None
        assert len(config.class_names) == 4
        assert config.input_shape == (128, 128, 3)

    def test_config_creates_directories_with_utils(self):
        """Test that config directory creation works with utils functions."""
        config = BaseConfig(working_dir=self.temp_dir)

        # Use utils to ensure directories exist
        ensure_directory_exists(config.models_dir)
        ensure_directory_exists(config.logs_dir)
        ensure_directory_exists(config.cache_dir)

        assert config.models_dir.exists()
        assert config.logs_dir.exists()
        assert config.cache_dir.exists()

    def test_module_structure(self):
        """Test that the module structure is properly organized."""
        import img_classifier_config
        import img_classifier_data

        # Verify __all__ exports
        assert "BaseConfig" in img_classifier_config.__all__
        assert "DatasetConfig" in img_classifier_config.__all__
        assert "BaseDataLoader" in img_classifier_data.__all__
        assert "ImageDataLoader" in img_classifier_data.__all__


@pytest.mark.integration
@pytest.mark.slow
class TestArchitectureGeneration:
    """Integration tests for architecture generation with real config."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        return

    def test_architecture_factory_with_dataset_config(self):
        """Test creating architectures using DatasetConfig."""
        pytest.importorskip("tensorflow", reason="TensorFlow not available")

        from img_classifier_models import ArchitectureFactory

        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            class_names=["class1", "class2", "class3", "class4"],
        )

        for complexity in ["simple", "medium", "deep"]:
            model = ArchitectureFactory.create(config, complexity=complexity)

            assert model is not None
            assert model.input_shape[1:] == config.input_shape
            assert model.output_shape[-1] == config.num_classes
