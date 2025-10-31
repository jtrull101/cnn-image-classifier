"""Tests for data loading modules."""

import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
from unittest.mock import MagicMock, patch

from src.alz_mri_cnn.config import BaseConfig
from src.alz_mri_cnn.data import BaseDataLoader, ImageDataLoader


class TestBaseDataLoader(unittest.TestCase):
    """Tests for BaseDataLoader abstract class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = BaseConfig(working_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that BaseDataLoader cannot be instantiated directly."""
        # Since it's abstract, we can't instantiate it
        # But we can test that it has the required methods
        self.assertTrue(hasattr(BaseDataLoader, 'load_train_data'))
        self.assertTrue(hasattr(BaseDataLoader, 'load_test_data'))
        self.assertTrue(hasattr(BaseDataLoader, 'download_dataset'))
        self.assertTrue(hasattr(BaseDataLoader, 'prepare_dataset'))

    def test_get_categories(self):
        """Test get_categories method."""
        # Create a concrete subclass for testing
        class ConcreteLoader(BaseDataLoader):
            def load_train_data(self):
                pass
            def load_test_data(self):
                pass
            def download_dataset(self):
                pass
            def prepare_dataset(self):
                pass

        loader = ConcreteLoader(self.config)
        
        # Create test directory structure
        test_path = self.temp_dir / "categories_test"
        test_path.mkdir()
        (test_path / "cat1").mkdir()
        (test_path / "cat2").mkdir()
        (test_path / ".hidden").mkdir()  # Should be ignored
        (test_path / "file.txt").write_text("test")  # Should be ignored
        
        categories = loader.get_categories(test_path)
        
        self.assertEqual(len(categories), 2)
        self.assertIn("cat1", categories)
        self.assertIn("cat2", categories)
        self.assertNotIn(".hidden", categories)

    def test_get_categories_empty_directory(self):
        """Test get_categories with empty directory."""
        class ConcreteLoader(BaseDataLoader):
            def load_train_data(self):
                pass
            def load_test_data(self):
                pass
            def download_dataset(self):
                pass
            def prepare_dataset(self):
                pass

        loader = ConcreteLoader(self.config)
        test_path = self.temp_dir / "empty"
        test_path.mkdir()
        
        categories = loader.get_categories(test_path)
        self.assertEqual(len(categories), 0)

    def test_get_categories_nonexistent_directory(self):
        """Test get_categories with non-existent directory."""
        class ConcreteLoader(BaseDataLoader):
            def load_train_data(self):
                pass
            def load_test_data(self):
                pass
            def download_dataset(self):
                pass
            def prepare_dataset(self):
                pass

        loader = ConcreteLoader(self.config)
        categories = loader.get_categories(self.temp_dir / "nonexistent")
        self.assertEqual(len(categories), 0)

    def test_split_data(self):
        """Test split_data method."""
        class ConcreteLoader(BaseDataLoader):
            def load_train_data(self):
                pass
            def load_test_data(self):
                pass
            def download_dataset(self):
                pass
            def prepare_dataset(self):
                pass

        loader = ConcreteLoader(self.config)
        
        # Create test data
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        
        X1, X2, y1, y2 = loader.split_data(X, y, split_ratio=0.7)
        
        self.assertEqual(len(X1), 70)
        self.assertEqual(len(X2), 30)
        self.assertEqual(len(y1), 70)
        self.assertEqual(len(y2), 30)

    def test_reduce_dataset(self):
        """Test reduce_dataset method."""
        class ConcreteLoader(BaseDataLoader):
            def load_train_data(self):
                pass
            def load_test_data(self):
                pass
            def download_dataset(self):
                pass
            def prepare_dataset(self):
                pass

        loader = ConcreteLoader(self.config)
        
        # Create test data
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        
        X_reduced, y_reduced = loader.reduce_dataset(X, y, percent=0.5)
        
        self.assertEqual(len(X_reduced), 50)
        self.assertEqual(len(y_reduced), 50)

    def test_reduce_dataset_full_size(self):
        """Test reduce_dataset with 100% returns original data."""
        class ConcreteLoader(BaseDataLoader):
            def load_train_data(self):
                pass
            def load_test_data(self):
                pass
            def download_dataset(self):
                pass
            def prepare_dataset(self):
                pass

        loader = ConcreteLoader(self.config)
        
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        
        X_reduced, y_reduced = loader.reduce_dataset(X, y, percent=1.0)
        
        self.assertEqual(len(X_reduced), 100)
        self.assertEqual(len(y_reduced), 100)


class TestImageDataLoader(unittest.TestCase):
    """Tests for ImageDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = BaseConfig(working_dir=self.temp_dir)
        self.loader = ImageDataLoader(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test ImageDataLoader initialization."""
        self.assertIsInstance(self.loader, BaseDataLoader)
        self.assertEqual(self.loader.config, self.config)

    def test_initialization_creates_directories(self):
        """Test that initialization creates required directories."""
        # Directories should be created during init
        self.assertTrue(self.config.working_dir.exists())
        self.assertTrue(self.config.data_path.exists())

    def test_download_dataset_no_id(self):
        """Test download_dataset with no dataset_zip_id."""
        result = self.loader.download_dataset()
        self.assertFalse(result)

    def test_prepare_dataset_no_zip(self):
        """Test prepare_dataset when zip file doesn't exist."""
        result = self.loader.prepare_dataset()
        self.assertFalse(result)

    def test_prepare_dataset_already_prepared(self):
        """Test prepare_dataset when dataset is already prepared."""
        # Create train and test directories with content
        self.config.train_path.mkdir(parents=True)
        self.config.test_path.mkdir(parents=True)
        (self.config.train_path / "class1").mkdir()
        (self.config.test_path / "class1").mkdir()
        
        result = self.loader.prepare_dataset()
        self.assertTrue(result)

    def test_get_cache_path(self):
        """Test _get_cache_path method."""
        x_path, y_path = self.loader._get_cache_path(train=True)
        
        self.assertEqual(x_path.name, "X_data_train.pkl")
        self.assertEqual(y_path.name, "y_data_train.pkl")
        self.assertEqual(x_path.parent, self.config.cache_dir)

    def test_load_from_cache_no_cache(self):
        """Test _load_from_cache when cache doesn't exist."""
        result = self.loader._load_from_cache(train=True)
        self.assertIsNone(result)

    def test_setup_validates_paths(self):
        """Test that setup validates dataset paths."""
        result = self.loader.setup()
        # Should return False since we don't have actual data
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
