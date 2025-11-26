"""Tests for data loading modules."""

import numpy as np
import pytest

from img_classifier_config import BaseConfig
from img_classifier_data import BaseDataLoader, ImageDataLoader


class TestBaseDataLoader:
    """Tests for BaseDataLoader abstract class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures."""
        self.temp_dir = tmp_path
        self.config = BaseConfig(working_dir=self.temp_dir)

        yield

    def test_initialization(self):
        """Test that BaseDataLoader cannot be instantiated directly."""
        # Since it's abstract, we can't instantiate it
        # But we can test that it has the required methods
        assert hasattr(BaseDataLoader, "load_train_data")
        assert hasattr(BaseDataLoader, "load_test_data")
        assert hasattr(BaseDataLoader, "download_dataset")
        assert hasattr(BaseDataLoader, "prepare_dataset")

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

        assert len(categories) == 2
        assert "cat1" in categories
        assert "cat2" in categories
        assert ".hidden" not in categories

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
        assert len(categories) == 0

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
        assert len(categories) == 0

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
        x = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        x1, x2, y1, y2 = loader.split_data(x, y, split_ratio=0.7)

        assert len(x1) == 70
        assert len(x2) == 30
        assert len(y1) == 70
        assert len(y2) == 30

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
        x = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        x_reduced, y_reduced = loader.reduce_dataset(x, y, percent=0.5)

        assert len(x_reduced) == 50
        assert len(y_reduced) == 50

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

        x = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        x_reduced, y_reduced = loader.reduce_dataset(x, y, percent=1.0)

        assert len(x_reduced) == 100
        assert len(y_reduced) == 100


class TestImageDataLoader:
    """Tests for ImageDataLoader class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures."""
        self.temp_dir = tmp_path
        self.config = BaseConfig(working_dir=self.temp_dir)
        self.loader = ImageDataLoader(self.config)

        yield

    def test_initialization(self):
        """Test ImageDataLoader initialization."""
        assert isinstance(self.loader, BaseDataLoader)
        assert self.loader.config == self.config

    def test_initialization_creates_directories(self):
        """Test that initialization creates required directories."""
        # Directories should be created during init
        assert self.config.working_dir.exists()
        assert self.config.data_path.exists()

    def test_download_dataset_no_id(self):
        """Test download_dataset with no dataset_zip_id."""
        result = self.loader.download_dataset()
        assert result is False

    def test_prepare_dataset_no_zip(self):
        """Test prepare_dataset when zip file doesn't exist."""
        result = self.loader.prepare_dataset()
        assert result is False

    def test_prepare_dataset_already_prepared(self):
        """Test prepare_dataset when dataset is already prepared."""
        # Create train and test directories with content
        self.config.train_path.mkdir(parents=True)
        self.config.test_path.mkdir(parents=True)
        (self.config.train_path / "class1").mkdir()
        (self.config.test_path / "class1").mkdir()

        result = self.loader.prepare_dataset()
        assert result is True

    def test_get_cache_path(self):
        """Test _get_cache_path method."""
        x_path, y_path = self.loader._get_cache_path(train=True)

        assert x_path.name == "X_data_train.pkl"
        assert y_path.name == "y_data_train.pkl"
        assert x_path.parent == self.config.cache_dir

    def test_load_from_cache_no_cache(self):
        """Test _load_from_cache when cache doesn't exist."""
        result = self.loader._load_from_cache(train=True)
        assert result is None

    def test_setup_validates_paths(self):
        """Test that setup validates dataset paths."""
        result = self.loader.setup()
        # Should return False since we don't have actual data
        assert result is False
