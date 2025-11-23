"""Integration tests for the rearchitected modules.

These tests verify that all modules work together correctly, including:
- Module import validation (ensuring all public APIs are accessible)
- Config and data loader integration (verifying components interact properly)
- Overall package structure validation
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from alz_mri_config import AlzheimerConfig


class TestArchitectureIntegration(unittest.TestCase):
    """Tests for overall architecture integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_config_imports(self):
        """Test that all config classes can be imported."""
        from alz_mri_config import AlzheimerConfig, BaseConfig
        self.assertIsNotNone(BaseConfig)
        self.assertIsNotNone(AlzheimerConfig)

    def test_data_loader_imports(self):
        """Test that all data loader classes can be imported."""
        from alz_mri_data import BaseDataLoader, ImageDataLoader
        self.assertIsNotNone(BaseDataLoader)
        self.assertIsNotNone(ImageDataLoader)

    def test_utils_imports(self):
        """Test that all utility functions can be imported."""
        from alz_mri_utils import (
            clean_directory,
            download_from_google_drive,
            ensure_directory_exists,
            extract_archive,
            organize_dataset,
        )
        self.assertIsNotNone(download_from_google_drive)
        self.assertIsNotNone(extract_archive)
        self.assertIsNotNone(organize_dataset)
        self.assertIsNotNone(clean_directory)
        self.assertIsNotNone(ensure_directory_exists)

    def test_config_and_data_loader_integration(self):
        """Test that config and data loader work together."""
        from alz_mri_config import BaseConfig
        from alz_mri_data import ImageDataLoader

        config = BaseConfig(working_dir=self.temp_dir)
        loader = ImageDataLoader(config)

        self.assertEqual(loader.config, config)
        self.assertTrue(config.working_dir.exists())

    def test_alzheimer_config_initialization(self):
        """Test that AlzheimerConfig initializes with all required fields."""
        config = AlzheimerConfig(working_dir=self.temp_dir)

        # Verify all expected fields are present
        self.assertEqual(config.num_classes, 4)
        self.assertIsNotNone(config.class_names)
        self.assertIsNotNone(config.nice_class_names)
        self.assertEqual(len(config.class_names), 4)
        self.assertEqual(len(config.nice_class_names), 4)

    def test_module_structure(self):
        """Test that the module structure is properly organized."""
        import alz_mri_config
        import alz_mri_data

        # Verify __all__ exports
        self.assertIn('BaseConfig', alz_mri_config.__all__)
        self.assertIn('AlzheimerConfig', alz_mri_config.__all__)
        self.assertIn('BaseDataLoader', alz_mri_data.__all__)
        self.assertIn('ImageDataLoader', alz_mri_data.__all__)


if __name__ == '__main__':
    unittest.main()
