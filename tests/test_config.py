"""Tests for configuration modules."""

from pathlib import Path

import pytest

from img_classifier_config import BaseConfig, DatasetConfig


class TestBaseConfig:
    """Tests for BaseConfig class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures."""
        self.temp_dir = str(tmp_path)
        self.config = BaseConfig(working_dir=Path(self.temp_dir))

        yield


    def test_default_values(self):
        """Test default configuration values."""
        config = BaseConfig()
        assert config.project_name == "mri_classifier"
        assert config.num_classes == 4
        assert config.batch_size == 32
        assert config.num_epochs == 25
        assert config.learning_rate == 0.001
        assert config.validation_split == 0.2
        assert config.dropout_rate == 0.3

    def test_image_size(self):
        """Test image size configuration."""
        config = BaseConfig()
        assert config.image_size == (128, 128)
        assert config.color_channels == 3

    def test_input_shape(self):
        """Test input_shape property."""
        config = BaseConfig()
        assert config.input_shape == (128, 128, 3)

    def test_custom_image_size(self):
        """Test custom image size."""
        config = BaseConfig(image_size=(64, 64))
        assert config.input_shape == (64, 64, 3)

    def test_paths_initialization(self):
        """Test that paths are initialized correctly."""
        assert isinstance(self.config.working_dir, Path)
        assert isinstance(self.config.data_path, Path)
        assert isinstance(self.config.train_path, Path)
        assert isinstance(self.config.test_path, Path)

    def test_derived_paths(self):
        """Test derived path properties."""
        assert self.config.models_dir == self.config.working_dir / "models"
        assert self.config.logs_dir == self.config.working_dir / "logs"
        assert self.config.cache_dir == self.config.data_path / "cache"

    def test_create_directories(self):
        """Test directory creation."""
        self.config.create_directories()

        assert self.config.working_dir.exists()
        assert self.config.data_path.exists()
        assert self.config.models_dir.exists()
        assert self.config.logs_dir.exists()
        assert self.config.cache_dir.exists()

    def test_string_to_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = BaseConfig(working_dir=Path("/tmp/test"))
        assert isinstance(config.working_dir, Path)

    def test_custom_paths(self):
        """Test custom path configuration."""
        data_path = Path(self.temp_dir) / "custom_data"
        config = BaseConfig(
            working_dir=Path(self.temp_dir),
            data_path=data_path
        )
        assert config.data_path == data_path


class TestDatasetConfig:
    """Tests for DatasetConfig class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures."""
        self.temp_dir = str(tmp_path)
        self.config = DatasetConfig(
            working_dir=Path(self.temp_dir),
            num_classes=4,
            class_names=["class1", "class2", "class3", "class4"]
        )

        yield


    def test_project_specific_values(self):
        """Test dataset-specific configuration."""
        config = DatasetConfig(
            project_name="my_classifier",
            dataset_name="my_dataset",
            working_dir=Path(self.temp_dir),
            num_classes=3,
            class_names=["cat", "dog", "bird"]
        )
        assert config.project_name == "my_classifier"
        assert config.dataset_name == "my_dataset"
        assert config.num_classes == 3

    def test_class_names(self):
        """Test that class names are set correctly."""
        assert self.config.num_classes == 4
        assert self.config.class_names is not None
        assert len(self.config.class_names) == 4

        expected_classes = ["class1", "class2", "class3", "class4"]
        assert self.config.class_names == expected_classes

    def test_architecture_complexity(self):
        """Test architecture complexity setting."""
        assert self.config.architecture_complexity in ["auto", "simple", "medium", "deep", "custom"]

    def test_inherits_base_config(self):
        """Test that DatasetConfig inherits from BaseConfig."""
        assert isinstance(self.config, BaseConfig)
        assert self.config.batch_size == 32
        assert self.config.num_epochs == 25

    def test_custom_class_names(self):
        """Test setting custom class names."""
        custom_names = ["Class1", "Class2", "Class3", "Class4"]
        config = DatasetConfig(
            working_dir=Path(self.temp_dir),
            num_classes=4,
            class_names=custom_names
        )
        assert config.class_names == custom_names

    def test_yaml_support(self):
        """Test YAML configuration support."""
        yaml_path = Path(self.temp_dir) / "test_config.yaml"
        self.config.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded_config = DatasetConfig.from_yaml(yaml_path)
        assert loaded_config.num_classes == self.config.num_classes
        assert loaded_config.class_names == self.config.class_names

    def test_validate_complexity_valid(self):
        """Test architecture complexity validation with valid values."""
        for complexity in ["auto", "simple", "medium", "deep", "custom"]:
            config = DatasetConfig(
                working_dir=self.temp_dir,
                architecture_complexity=complexity
            )
            assert config.architecture_complexity == complexity

    def test_validate_complexity_invalid(self):
        """Test architecture complexity validation with invalid value."""
        with pytest.raises(ValueError, match="architecture_complexity must be one of"):
            DatasetConfig(
                working_dir=self.temp_dir,
                architecture_complexity="invalid"
            )

    def test_recommended_batch_sizes_default(self):
        """Test default recommended batch sizes."""
        config = DatasetConfig(working_dir=self.temp_dir)
        assert config.recommended_batch_sizes == [16, 32, 64]

    def test_recommended_learning_rates_default(self):
        """Test default recommended learning rates."""
        config = DatasetConfig(working_dir=self.temp_dir)
        assert config.recommended_learning_rates == [0.0001, 0.001, 0.01]

    def test_dataset_type_default(self):
        """Test default dataset type."""
        config = DatasetConfig(working_dir=self.temp_dir)
        assert config.dataset_type == "image_classification"

    def test_auto_detect_classes_default(self):
        """Test default auto_detect_classes setting."""
        config = DatasetConfig(working_dir=self.temp_dir)
        assert config.auto_detect_classes is True

    def test_min_images_per_class_default(self):
        """Test default min_images_per_class setting."""
        config = DatasetConfig(working_dir=self.temp_dir)
        assert config.min_images_per_class == 10

    def test_yaml_with_nested_paths(self):
        """Test YAML save/load with nested directory structure."""
        nested_yaml = Path(self.temp_dir) / "configs" / "nested" / "config.yaml"
        self.config.to_yaml(nested_yaml)
        assert nested_yaml.exists()
        assert nested_yaml.parent.exists()

        loaded = DatasetConfig.from_yaml(nested_yaml)
        assert loaded.working_dir == self.config.working_dir

    def test_yaml_path_conversion(self):
        """Test that paths are properly converted in YAML."""
        yaml_path = Path(self.temp_dir) / "path_test.yaml"
        self.config.to_yaml(yaml_path)

        # Read raw YAML to check string conversion
        with open(yaml_path, 'r') as f:
            raw_yaml = f.read()

        # Paths should be strings in YAML
        assert str(self.config.working_dir) in raw_yaml

    def test_description_field(self):
        """Test optional description field."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            description="Test dataset for unit testing"
        )
        assert config.description == "Test dataset for unit testing"

    def test_custom_recommended_settings(self):
        """Test custom recommended batch sizes and learning rates."""
        custom_batches = [8, 16, 32]
        custom_lrs = [0.00001, 0.0001]

        config = DatasetConfig(
            working_dir=self.temp_dir,
            recommended_batch_sizes=custom_batches,
            recommended_learning_rates=custom_lrs
        )

        assert config.recommended_batch_sizes == custom_batches
        assert config.recommended_learning_rates == custom_lrs
