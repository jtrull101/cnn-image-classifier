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


class TestBaseConfigEdgeCases:
    """Edge case tests for BaseConfig."""

    def test_color_channels_boundary_min(self):
        """Test minimum color channel boundary (1)."""
        config = BaseConfig(color_channels=1)
        assert config.color_channels == 1
        assert config.input_shape == (128, 128, 1)

    def test_color_channels_boundary_max(self):
        """Test maximum color channel boundary (4)."""
        config = BaseConfig(color_channels=4)
        assert config.color_channels == 4
        assert config.input_shape == (128, 128, 4)

    def test_num_classes_boundary_min(self):
        """Test minimum num_classes boundary (2)."""
        config = BaseConfig(num_classes=2)
        assert config.num_classes == 2

    def test_batch_size_boundary_min(self):
        """Test minimum batch size boundary (1)."""
        config = BaseConfig(batch_size=1)
        assert config.batch_size == 1

    def test_num_epochs_boundary_min(self):
        """Test minimum num_epochs boundary (1)."""
        config = BaseConfig(num_epochs=1)
        assert config.num_epochs == 1

    def test_learning_rate_boundary_max(self):
        """Test maximum learning rate boundary (1.0)."""
        config = BaseConfig(learning_rate=1.0)
        assert config.learning_rate == 1.0

    def test_validation_split_boundary_min(self):
        """Test minimum validation split boundary (0.0)."""
        config = BaseConfig(validation_split=0.0)
        assert config.validation_split == 0.0

    def test_validation_split_boundary_near_max(self):
        """Test validation split near maximum (0.99)."""
        config = BaseConfig(validation_split=0.99)
        assert config.validation_split == 0.99

    def test_test_split_boundary_near_max(self):
        """Test test split near maximum (0.99)."""
        config = BaseConfig(test_split=0.99)
        assert config.test_split == 0.99

    def test_dropout_rate_boundary_min(self):
        """Test minimum dropout rate boundary (0.0)."""
        config = BaseConfig(dropout_rate=0.0)
        assert config.dropout_rate == 0.0

    def test_dropout_rate_boundary_near_max(self):
        """Test dropout rate near maximum (0.99)."""
        config = BaseConfig(dropout_rate=0.99)
        assert config.dropout_rate == 0.99

    def test_data_percent_boundary_near_min(self):
        """Test data_percent near minimum (0.01)."""
        config = BaseConfig(data_percent=0.01)
        assert config.data_percent == 0.01

    def test_data_percent_boundary_max(self):
        """Test data_percent at maximum (1.0)."""
        config = BaseConfig(data_percent=1.0)
        assert config.data_percent == 1.0

    def test_early_stopping_patience_boundary_min(self):
        """Test minimum early stopping patience (1)."""
        config = BaseConfig(early_stopping_patience=1)
        assert config.early_stopping_patience == 1

    def test_min_accuracy_to_save_boundary_min(self):
        """Test minimum accuracy to save (0.0)."""
        config = BaseConfig(min_accuracy_to_save=0.0)
        assert config.min_accuracy_to_save == 0.0

    def test_min_accuracy_to_save_boundary_max(self):
        """Test maximum accuracy to save (1.0)."""
        config = BaseConfig(min_accuracy_to_save=1.0)
        assert config.min_accuracy_to_save == 1.0

    def test_disable_all_callbacks(self):
        """Test disabling all callback flags."""
        config = BaseConfig(
            use_early_stopping=False,
            use_checkpointing=False,
            use_model_checkpoint=False,
            use_accuracy_threshold_stopping=False
        )
        assert config.use_early_stopping is False
        assert config.use_checkpointing is False
        assert config.use_model_checkpoint is False
        assert config.use_accuracy_threshold_stopping is False

    def test_accuracy_threshold_boundary(self):
        """Test accuracy threshold boundary value."""
        config = BaseConfig(accuracy_threshold=0.999)
        assert config.accuracy_threshold == 0.999

    def test_custom_dataset_zip_id(self):
        """Test setting custom dataset zip ID."""
        config = BaseConfig(dataset_zip_id="1234567890abcdef")
        assert config.dataset_zip_id == "1234567890abcdef"

    def test_custom_dataset_name(self):
        """Test setting custom dataset name."""
        config = BaseConfig(dataset_name="my_custom_dataset")
        assert config.dataset_name == "my_custom_dataset"

    def test_empty_class_names_list(self):
        """Test empty class names list."""
        config = BaseConfig(class_names=[])
        assert config.class_names == []

    def test_large_num_classes(self):
        """Test large number of classes."""
        config = BaseConfig(num_classes=1000)
        assert config.num_classes == 1000

    def test_large_batch_size(self):
        """Test large batch size."""
        config = BaseConfig(batch_size=512)
        assert config.batch_size == 512

    def test_large_num_epochs(self):
        """Test large number of epochs."""
        config = BaseConfig(num_epochs=1000)
        assert config.num_epochs == 1000

    def test_very_small_learning_rate(self):
        """Test very small learning rate."""
        config = BaseConfig(learning_rate=0.0000001)
        assert config.learning_rate == 0.0000001

    def test_custom_image_size_rectangular(self):
        """Test rectangular image size."""
        config = BaseConfig(image_size=(256, 128))
        assert config.image_size == (256, 128)
        assert config.input_shape == (256, 128, 3)

    def test_very_small_image_size(self):
        """Test very small image size."""
        config = BaseConfig(image_size=(32, 32))
        assert config.image_size == (32, 32)
        assert config.input_shape == (32, 32, 3)

    def test_very_large_image_size(self):
        """Test very large image size."""
        config = BaseConfig(image_size=(512, 512))
        assert config.image_size == (512, 512)
        assert config.input_shape == (512, 512, 3)

    def test_paths_with_none_data_path(self, tmp_path):
        """Test that paths are created properly when data_path is None."""
        config = BaseConfig(working_dir=tmp_path, data_path=None)
        assert config.data_path == tmp_path / "data" / "dataset"
        assert config.train_path == config.data_path / "train"
        assert config.test_path == config.data_path / "test"

    def test_paths_with_explicit_train_test(self, tmp_path):
        """Test explicit train and test paths."""
        train = tmp_path / "custom_train"
        test = tmp_path / "custom_test"
        config = BaseConfig(
            working_dir=tmp_path,
            train_path=train,
            test_path=test
        )
        assert config.train_path == train
        assert config.test_path == test

    def test_create_directories_idempotent(self, tmp_path):
        """Test that create_directories can be called multiple times."""
        config = BaseConfig(working_dir=tmp_path)
        config.create_directories()
        config.create_directories()
        assert config.working_dir.exists()
        assert config.models_dir.exists()


class TestDatasetConfigEdgeCases:
    """Edge case tests for DatasetConfig."""

    def test_from_yaml_nonexistent_file(self, tmp_path):
        """Test loading from non-existent YAML file."""
        yaml_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            DatasetConfig.from_yaml(yaml_path)

    def test_to_yaml_creates_parent_dirs(self, tmp_path):
        """Test that to_yaml creates parent directories."""
        config = DatasetConfig(working_dir=tmp_path)
        nested_path = tmp_path / "level1" / "level2" / "level3" / "config.yaml"
        config.to_yaml(nested_path)
        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_yaml_with_none_values(self, tmp_path):
        """Test YAML serialization with None values."""
        config = DatasetConfig(
            working_dir=tmp_path,
            description=None,
            dataset_zip_id=None
        )
        yaml_path = tmp_path / "test.yaml"
        config.to_yaml(yaml_path)

        loaded = DatasetConfig.from_yaml(yaml_path)
        assert loaded.description is None

    def test_yaml_roundtrip_with_all_fields(self, tmp_path):
        """Test YAML roundtrip with all fields set."""
        config = DatasetConfig(
            working_dir=tmp_path,
            project_name="test_project",
            dataset_name="test_dataset",
            dataset_type="image_classification",
            description="Test description",
            num_classes=5,
            class_names=["a", "b", "c", "d", "e"],
            architecture_complexity="deep",
            recommended_batch_sizes=[8, 16],
            recommended_learning_rates=[0.0001, 0.001],
            auto_detect_classes=False,
            min_images_per_class=20,
            batch_size=64,
            num_epochs=50,
            learning_rate=0.0005,
            validation_split=0.15,
            dropout_rate=0.5
        )

        yaml_path = tmp_path / "full_config.yaml"
        config.to_yaml(yaml_path)
        loaded = DatasetConfig.from_yaml(yaml_path)

        assert loaded.project_name == config.project_name
        assert loaded.dataset_name == config.dataset_name
        assert loaded.description == config.description
        assert loaded.num_classes == config.num_classes
        assert loaded.class_names == config.class_names
        assert loaded.architecture_complexity == config.architecture_complexity
        assert loaded.batch_size == config.batch_size
        assert loaded.learning_rate == config.learning_rate

    def test_invalid_architecture_complexity_empty_string(self, tmp_path):
        """Test empty string for architecture complexity."""
        with pytest.raises(ValueError, match="architecture_complexity must be one of"):
            DatasetConfig(
                working_dir=tmp_path,
                architecture_complexity=""
            )

    def test_all_valid_architecture_complexities(self, tmp_path):
        """Test all valid architecture complexity values."""
        valid_complexities = ["auto", "simple", "medium", "deep", "custom"]
        for complexity in valid_complexities:
            config = DatasetConfig(
                working_dir=tmp_path,
                architecture_complexity=complexity
            )
            assert config.architecture_complexity == complexity

    def test_empty_class_names_with_zero_classes(self, tmp_path):
        """Test edge case with mismatched class_names and num_classes."""
        config = DatasetConfig(
            working_dir=tmp_path,
            num_classes=5,
            class_names=[]  # Empty list but num_classes is 5
        )
        assert config.num_classes == 5
        assert len(config.class_names) == 0

    def test_min_images_per_class_edge_values(self, tmp_path):
        """Test min_images_per_class with edge values."""
        config1 = DatasetConfig(working_dir=tmp_path, min_images_per_class=1)
        assert config1.min_images_per_class == 1

        config2 = DatasetConfig(working_dir=tmp_path, min_images_per_class=10000)
        assert config2.min_images_per_class == 10000

    def test_dataset_type_custom_value(self, tmp_path):
        """Test custom dataset type."""
        config = DatasetConfig(
            working_dir=tmp_path,
            dataset_type="medical_imaging"
        )
        assert config.dataset_type == "medical_imaging"

    def test_long_description(self, tmp_path):
        """Test very long description field."""
        long_desc = "A" * 10000
        config = DatasetConfig(
            working_dir=tmp_path,
            description=long_desc
        )
        assert config.description == long_desc

    def test_special_characters_in_class_names(self, tmp_path):
        """Test class names with special characters."""
        special_names = ["class-1", "class_2", "class.3", "class 4"]
        config = DatasetConfig(
            working_dir=tmp_path,
            num_classes=4,
            class_names=special_names
        )
        assert config.class_names == special_names

    def test_unicode_in_class_names(self, tmp_path):
        """Test class names with unicode characters."""
        unicode_names = ["类别1", "κλάση2", "クラス3"]
        config = DatasetConfig(
            working_dir=tmp_path,
            num_classes=3,
            class_names=unicode_names
        )
        assert config.class_names == unicode_names

    def test_empty_recommended_lists(self, tmp_path):
        """Test empty recommended batch sizes and learning rates."""
        config = DatasetConfig(
            working_dir=tmp_path,
            recommended_batch_sizes=[],
            recommended_learning_rates=[]
        )
        assert config.recommended_batch_sizes == []
        assert config.recommended_learning_rates == []

    def test_single_element_recommended_lists(self, tmp_path):
        """Test single element in recommended lists."""
        config = DatasetConfig(
            working_dir=tmp_path,
            recommended_batch_sizes=[32],
            recommended_learning_rates=[0.001]
        )
        assert config.recommended_batch_sizes == [32]
        assert config.recommended_learning_rates == [0.001]

    def test_many_recommended_values(self, tmp_path):
        """Test many recommended values."""
        config = DatasetConfig(
            working_dir=tmp_path,
            recommended_batch_sizes=[4, 8, 16, 32, 64, 128, 256],
            recommended_learning_rates=[0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01]
        )
        assert len(config.recommended_batch_sizes) == 7
        assert len(config.recommended_learning_rates) == 6


class TestAlzheimerConfig:
    """Tests for AlzheimerConfig class."""

    def test_alzheimer_config_defaults(self):
        """Test Alzheimer's config default values."""
        from img_classifier_config import AlzheimerConfig

        config = AlzheimerConfig()
        assert config.project_name == "alzheimer_mri_cnn"
        assert config.num_classes == 4
        assert len(config.class_names) == 4
        assert len(config.nice_class_names) == 4

    def test_alzheimer_class_names(self):
        """Test Alzheimer's specific class names."""
        from img_classifier_config import AlzheimerConfig

        config = AlzheimerConfig()
        expected_names = [
            "MildDemented",
            "NonDemented",
            "ModerateDemented",
            "VeryMildDemented",
        ]
        assert config.class_names == expected_names

    def test_alzheimer_nice_class_names(self):
        """Test Alzheimer's nice class names."""
        from img_classifier_config import AlzheimerConfig

        config = AlzheimerConfig()
        expected_nice = [
            "Mild Impairment",
            "No Impairment",
            "Moderate Impairment",
            "Very Mild Impairment",
        ]
        assert config.nice_class_names == expected_nice

    def test_alzheimer_dataset_info(self):
        """Test Alzheimer's dataset specific info."""
        from img_classifier_config import AlzheimerConfig

        config = AlzheimerConfig()
        assert config.dataset_name == "Combined Dataset"
        assert config.dataset_zip_id == "1SQuB_8IL3s7vZPMeGkOZo116QSTMa6BN"
        assert config.pretrained_model_id == "1U9uywbNatIFAj6XlahT6BBrMqyLgd4qZ"

    def test_alzheimer_inherits_base_config(self):
        """Test that AlzheimerConfig inherits from BaseConfig."""
        from img_classifier_config import AlzheimerConfig

        config = AlzheimerConfig()
        assert isinstance(config, BaseConfig)
        assert hasattr(config, "batch_size")
        assert hasattr(config, "num_epochs")
        assert hasattr(config, "input_shape")

    def test_alzheimer_config_override(self, tmp_path):
        """Test overriding AlzheimerConfig values."""
        from img_classifier_config import AlzheimerConfig

        config = AlzheimerConfig(
            working_dir=tmp_path,
            batch_size=64,
            num_epochs=50
        )
        assert config.working_dir == tmp_path
        assert config.batch_size == 64
        assert config.num_epochs == 50
        # Should still have Alzheimer-specific values
        assert config.num_classes == 4
        assert len(config.class_names) == 4


class TestDatasetDetector:
    """Tests for DatasetDetector class."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock dataset structure."""
        from img_classifier_config.dataset_config import DatasetDetector

        # Create train/test split structure
        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"
        train_dir.mkdir()
        test_dir.mkdir()

        # Create class directories
        for split_dir in [train_dir, test_dir]:
            for class_name in ["cat", "dog", "bird"]:
                class_dir = split_dir / class_name
                class_dir.mkdir()
                # Create dummy image files
                for i in range(5):
                    (class_dir / f"image_{i}.jpg").touch()

        return tmp_path

    def test_detector_initialization(self, tmp_path):
        """Test DatasetDetector initialization."""
        from img_classifier_config.dataset_config import DatasetDetector

        detector = DatasetDetector(tmp_path)
        assert detector.dataset_path == tmp_path

    def test_detect_nonexistent_path(self, tmp_path):
        """Test detection with non-existent path."""
        from img_classifier_config.dataset_config import DatasetDetector

        nonexistent = tmp_path / "nonexistent"
        detector = DatasetDetector(nonexistent)
        with pytest.raises(ValueError, match="Dataset path does not exist"):
            detector.detect()

    def test_detect_train_test_split(self, mock_dataset):
        """Test detection of train/test split."""
        from img_classifier_config.dataset_config import DatasetDetector

        detector = DatasetDetector(mock_dataset)
        info = detector.detect()

        assert info["has_train_test_split"] is True
        assert info["num_classes"] == 3
        assert set(info["class_names"]) == {"cat", "dog", "bird"}

    def test_detect_no_train_test_split(self, tmp_path):
        """Test detection without train/test split."""
        from img_classifier_config.dataset_config import DatasetDetector

        # Create flat structure
        for class_name in ["class1", "class2"]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(10):
                (class_dir / f"img_{i}.png").touch()

        detector = DatasetDetector(tmp_path)
        info = detector.detect()

        assert info["has_train_test_split"] is False
        assert info["num_classes"] == 2
        assert info["total_images"] == 20

    def test_detect_class_distribution(self, tmp_path):
        """Test class distribution detection."""
        from img_classifier_config.dataset_config import DatasetDetector

        # Create imbalanced dataset
        for class_name, count in [("a", 100), ("b", 50), ("c", 25)]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(count):
                (class_dir / f"img_{i}.jpg").touch()

        detector = DatasetDetector(tmp_path)
        info = detector.detect()

        assert info["class_distribution"]["a"] == 100
        assert info["class_distribution"]["b"] == 50
        assert info["class_distribution"]["c"] == 25
        assert info["total_images"] == 175

    def test_check_balance_balanced(self, tmp_path):
        """Test balance check with balanced dataset."""
        from img_classifier_config.dataset_config import DatasetDetector

        # Create balanced dataset
        for class_name in ["a", "b", "c"]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(100):
                (class_dir / f"img_{i}.jpg").touch()

        detector = DatasetDetector(tmp_path)
        info = detector.detect()

        assert info["is_balanced"] is True

    def test_check_balance_imbalanced(self, tmp_path):
        """Test balance check with imbalanced dataset."""
        from img_classifier_config.dataset_config import DatasetDetector

        # Create imbalanced dataset (ratio > 2:1)
        for class_name, count in [("a", 100), ("b", 30)]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(count):
                (class_dir / f"img_{i}.jpg").touch()

        detector = DatasetDetector(tmp_path)
        info = detector.detect()

        assert info["is_balanced"] is False

    def test_recommend_complexity_simple(self, tmp_path):
        """Test complexity recommendation for small dataset."""
        from img_classifier_config.dataset_config import DatasetDetector

        # Small dataset
        for class_name in ["a", "b"]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(50):
                (class_dir / f"img_{i}.jpg").touch()

        detector = DatasetDetector(tmp_path)
        info = detector.detect()

        assert info["recommended_complexity"] == "simple"

    def test_recommend_complexity_medium(self, tmp_path):
        """Test complexity recommendation for medium dataset."""
        from img_classifier_config.dataset_config import DatasetDetector

        # Medium dataset
        for class_name in ["a", "b", "c", "d"]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(300):
                (class_dir / f"img_{i}.jpg").touch()

        detector = DatasetDetector(tmp_path)
        info = detector.detect()

        assert info["recommended_complexity"] == "medium"

    def test_supported_image_extensions(self, tmp_path):
        """Test detection of various image extensions."""
        from img_classifier_config.dataset_config import DatasetDetector

        class_dir = tmp_path / "class1"
        class_dir.mkdir()

        # Create files with different extensions
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
        for i, ext in enumerate(extensions):
            (class_dir / f"image_{i}{ext}").touch()

        # Create non-image file
        (class_dir / "readme.txt").touch()

        detector = DatasetDetector(tmp_path)
        info = detector.detect()

        assert info["total_images"] == len(extensions)

    def test_detect_with_mixed_case_extensions(self, tmp_path):
        """Test detection with mixed case extensions."""
        from img_classifier_config.dataset_config import DatasetDetector

        class_dir = tmp_path / "class1"
        class_dir.mkdir()

        # Create files with mixed case extensions
        (class_dir / "image1.JPG").touch()
        (class_dir / "image2.Png").touch()
        (class_dir / "image3.JPEG").touch()

        detector = DatasetDetector(tmp_path)
        info = detector.detect()

        assert info["total_images"] == 3

    def test_detect_empty_class_directories(self, tmp_path):
        """Test detection with empty class directories."""
        from img_classifier_config.dataset_config import DatasetDetector

        # Create directories but no images
        (tmp_path / "empty_class").mkdir()
        (tmp_path / "another_empty").mkdir()

        # Create one with images
        class_dir = tmp_path / "valid_class"
        class_dir.mkdir()
        (class_dir / "image.jpg").touch()

        detector = DatasetDetector(tmp_path)
        info = detector.detect()

        # Should only detect the class with images
        assert info["num_classes"] == 1
        assert "valid_class" in info["class_names"]

    def test_create_config_from_detector(self, tmp_path):
        """Test creating config from detector."""
        from img_classifier_config.dataset_config import DatasetDetector

        # Create dataset
        for class_name in ["a", "b", "c"]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(50):
                (class_dir / f"img_{i}.jpg").touch()

        detector = DatasetDetector(tmp_path)
        config = detector.create_config(
            project_name="test_project",
            working_dir=tmp_path / "work"
        )

        assert config.num_classes == 3
        assert set(config.class_names) == {"a", "b", "c"}
        assert config.project_name == "test_project"
        assert config.architecture_complexity == "simple"

    def test_create_config_with_overrides(self, tmp_path):
        """Test creating config with custom overrides."""
        from img_classifier_config.dataset_config import DatasetDetector

        # Create dataset
        class_dir = tmp_path / "class1"
        class_dir.mkdir()
        (class_dir / "img.jpg").touch()

        detector = DatasetDetector(tmp_path)
        config = detector.create_config(
            batch_size=128,
            learning_rate=0.0001,
            num_epochs=100
        )

        assert config.batch_size == 128
        assert config.learning_rate == 0.0001
        assert config.num_epochs == 100

    def test_detect_mismatched_train_test_classes(self, tmp_path):
        """Test detection when train and test have different classes."""
        from img_classifier_config.dataset_config import DatasetDetector

        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"
        train_dir.mkdir()
        test_dir.mkdir()

        # Create different classes in train and test
        for class_name in ["a", "b", "c"]:
            (train_dir / class_name).mkdir()
            (train_dir / class_name / "img.jpg").touch()

        for class_name in ["b", "c", "d"]:
            (test_dir / class_name).mkdir()
            (test_dir / class_name / "img.jpg").touch()

        detector = DatasetDetector(tmp_path)
        info = detector.detect()

        # Should include all classes
        assert info["num_classes"] == 4
        assert set(info["class_names"]) == {"a", "b", "c", "d"}

