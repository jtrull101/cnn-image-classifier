"""Tests for configuration modules."""

import shutil
import tempfile
import unittest
from pathlib import Path

from alz_mri_config import AlzheimerConfig, BaseConfig


class TestBaseConfig(unittest.TestCase):
    """Tests for BaseConfig class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = BaseConfig(working_dir=Path(self.temp_dir))

    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_default_values(self):
        """Test default configuration values."""
        config = BaseConfig()
        self.assertEqual(config.project_name, "mri_classifier")
        self.assertEqual(config.num_classes, 4)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.num_epochs, 25)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.validation_split, 0.2)
        self.assertEqual(config.dropout_rate, 0.3)

    def test_image_size(self):
        """Test image size configuration."""
        config = BaseConfig()
        self.assertEqual(config.image_size, (128, 128))
        self.assertEqual(config.color_channels, 3)

    def test_input_shape(self):
        """Test input_shape property."""
        config = BaseConfig()
        self.assertEqual(config.input_shape, (128, 128, 3))

    def test_custom_image_size(self):
        """Test custom image size."""
        config = BaseConfig(image_size=(64, 64))
        self.assertEqual(config.input_shape, (64, 64, 3))

    def test_paths_initialization(self):
        """Test that paths are initialized correctly."""
        self.assertIsInstance(self.config.working_dir, Path)
        self.assertIsInstance(self.config.data_path, Path)
        self.assertIsInstance(self.config.train_path, Path)
        self.assertIsInstance(self.config.test_path, Path)

    def test_derived_paths(self):
        """Test derived path properties."""
        self.assertEqual(
            self.config.models_dir,
            self.config.working_dir / "models"
        )
        self.assertEqual(
            self.config.logs_dir,
            self.config.working_dir / "logs"
        )
        self.assertEqual(
            self.config.cache_dir,
            self.config.data_path / "cache"
        )

    def test_create_directories(self):
        """Test directory creation."""
        self.config.create_directories()

        self.assertTrue(self.config.working_dir.exists())
        self.assertTrue(self.config.data_path.exists())
        self.assertTrue(self.config.models_dir.exists())
        self.assertTrue(self.config.logs_dir.exists())
        self.assertTrue(self.config.cache_dir.exists())

    def test_string_to_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = BaseConfig(working_dir="/tmp/test")
        self.assertIsInstance(config.working_dir, Path)

    def test_custom_paths(self):
        """Test custom path configuration."""
        data_path = Path(self.temp_dir) / "custom_data"
        config = BaseConfig(
            working_dir=Path(self.temp_dir),
            data_path=data_path
        )
        self.assertEqual(config.data_path, data_path)


class TestAlzheimerConfig(unittest.TestCase):
    """Tests for AlzheimerConfig class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AlzheimerConfig(working_dir=Path(self.temp_dir))

    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_project_specific_values(self):
        """Test Alzheimer-specific configuration."""
        self.assertEqual(self.config.project_name, "alzheimer_mri_cnn")
        self.assertEqual(self.config.dataset_name, "Combined Dataset")
        self.assertIsNotNone(self.config.dataset_zip_id)
        self.assertIsNotNone(self.config.pretrained_model_id)

    def test_class_names(self):
        """Test that class names are set correctly."""
        self.assertEqual(self.config.num_classes, 4)
        self.assertIsNotNone(self.config.class_names)
        self.assertEqual(len(self.config.class_names), 4)

        expected_classes = [
            "MildDemented",
            "NonDemented",
            "ModerateDemented",
            "VeryMildDemented",
        ]
        self.assertEqual(self.config.class_names, expected_classes)

    def test_nice_class_names(self):
        """Test that nice class names are set correctly."""
        self.assertIsNotNone(self.config.nice_class_names)
        self.assertEqual(len(self.config.nice_class_names), 4)

        expected_nice_names = [
            "Mild Impairment",
            "No Impairment",
            "Moderate Impairment",
            "Very Mild Impairment",
        ]
        self.assertEqual(self.config.nice_class_names, expected_nice_names)

    def test_inherits_base_config(self):
        """Test that AlzheimerConfig inherits from BaseConfig."""
        self.assertIsInstance(self.config, BaseConfig)
        self.assertEqual(self.config.batch_size, 32)
        self.assertEqual(self.config.num_epochs, 25)

    def test_custom_class_names(self):
        """Test setting custom class names."""
        custom_names = ["Class1", "Class2", "Class3", "Class4"]
        config = AlzheimerConfig(
            working_dir=Path(self.temp_dir),
            class_names=custom_names
        )
        self.assertEqual(config.class_names, custom_names)


if __name__ == '__main__':
    unittest.main()
