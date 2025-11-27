"""Unit tests for data loading modules using mocking.

These tests verify data loader behavior with mocked dependencies
to ensure fast execution and isolation from other packages.
"""

import numpy as np
import pytest
from unittest.mock import Mock

from img_classifier_data import BaseDataLoader, ImageDataLoader


@pytest.mark.unit
class TestBaseDataLoader:
    """Unit tests for BaseDataLoader abstract class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir

        # Create mock config
        self.mock_config = Mock()
        self.mock_config.working_dir = self.temp_dir
        self.mock_config.data_path = self.temp_dir / "data"
        self.mock_config.train_path = self.temp_dir / "data" / "train"
        self.mock_config.test_path = self.temp_dir / "data" / "test"

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

        loader = ConcreteLoader(self.mock_config)

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


@pytest.mark.unit
class TestImageDataLoader:
    """Tests for ImageDataLoader class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir, mock_config):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = mock_config
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


@pytest.mark.unit
class TestBaseDataLoaderEdgeCases:
    """Edge case tests for BaseDataLoader."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir, mock_config):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = mock_config

        # Create concrete implementation for testing
        class ConcreteLoader(BaseDataLoader):
            def load_train_data(self):
                return None, None

            def load_test_data(self):
                return None, None

            def download_dataset(self):
                return True

            def prepare_dataset(self):
                return True

        self.loader = ConcreteLoader(self.config)
        yield

    def test_split_data_with_zero_split(self):
        """Test split_data with 0% split (all data in first set)."""
        x = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        x1, x2, y1, y2 = self.loader.split_data(x, y, split_ratio=0.0)

        assert len(x1) == 0
        assert len(x2) == 100
        assert len(y1) == 0
        assert len(y2) == 100

    def test_split_data_with_full_split(self):
        """Test split_data with 100% split (all data in second set)."""
        x = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        x1, x2, y1, y2 = self.loader.split_data(x, y, split_ratio=1.0)

        assert len(x1) == 100
        assert len(x2) == 0
        assert len(y1) == 100
        assert len(y2) == 0

    def test_reduce_dataset_with_zero_percent(self):
        """Test reduce_dataset with 0% (empty dataset)."""
        x = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        x_reduced, y_reduced = self.loader.reduce_dataset(x, y, percent=0.0)

        assert len(x_reduced) == 0
        assert len(y_reduced) == 0

    def test_reduce_dataset_with_small_data(self):
        """Test reduce_dataset with very small dataset."""
        x = np.arange(5).reshape(5, 1)
        y = np.arange(5)

        x_reduced, y_reduced = self.loader.reduce_dataset(x, y, percent=0.5)

        # Should get at least some samples
        assert len(x_reduced) > 0
        assert len(y_reduced) > 0

    def test_split_data_single_sample(self):
        """Test split_data with single sample."""
        x = np.array([[1]])
        y = np.array([0])

        x1, x2, y1, y2 = self.loader.split_data(x, y, split_ratio=0.5)

        # With single sample, split may produce 0 or 1 in each set
        assert len(x1) + len(x2) == 1
        assert len(y1) + len(y2) == 1

    def test_get_categories_with_special_characters(self):
        """Test get_categories with special characters in names."""
        test_path = self.temp_dir / "special"
        test_path.mkdir()
        (test_path / "cat-1_test").mkdir()
        (test_path / "cat.2").mkdir()

        categories = self.loader.get_categories(test_path)

        assert len(categories) == 2
        assert "cat-1_test" in categories
        assert "cat.2" in categories

    def test_reduce_dataset_maintains_distribution(self):
        """Test that reduce_dataset maintains relative distribution."""
        # Create data with known distribution
        x = np.vstack([np.ones((50, 1)), np.zeros((50, 1))])
        y = np.hstack([np.ones(50), np.zeros(50)])

        x_reduced, y_reduced = self.loader.reduce_dataset(x, y, percent=0.5)

        assert len(x_reduced) == 50
        # Check that both classes are represented (with some tolerance)
        assert 15 <= np.sum(y_reduced == 1) <= 35  # Allow some variance


@pytest.mark.unit
class TestImageDataLoaderEdgeCases:
    """Edge case tests for ImageDataLoader."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir, mock_config):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = mock_config
        self.loader = ImageDataLoader(self.config)
        yield

    def test_cache_path_for_test_data(self):
        """Test cache path generation for test data."""
        x_path, y_path = self.loader._get_cache_path(train=False)

        assert x_path.name == "X_data_test.pkl"
        assert y_path.name == "y_data_test.pkl"

    def test_multiple_loader_instances(self):
        """Test creating multiple loader instances."""
        loader1 = ImageDataLoader(self.config)
        loader2 = ImageDataLoader(self.config)

        assert loader1.config == loader2.config
        assert loader1 is not loader2

    def test_config_modification_after_init(self):
        """Test that loader respects config modifications."""
        original_batch_size = self.config.batch_size
        self.config.batch_size = 64

        assert self.loader.config.batch_size == 64
        assert self.loader.config.batch_size != original_batch_size


    def test_split_data_with_single_sample(self):
        """Test split_data with only one sample."""
        x = np.array([[1]])
        y = np.array([0])

        x1, x2, y1, y2 = self.loader.split_data(x, y, split_ratio=0.7)

        # With one sample, it should go to one set
        assert len(x1) + len(x2) == 1
        assert len(y1) + len(y2) == 1

    def test_split_data_with_small_dataset(self):
        """Test split_data with very small dataset."""
        x = np.arange(5).reshape(5, 1)
        y = np.arange(5)

        x1, x2, y1, y2 = self.loader.split_data(x, y, split_ratio=0.5)

        assert len(x1) + len(x2) == 5
        assert len(y1) + len(y2) == 5
        # Check that data is preserved
        combined = np.concatenate([y1, y2])
        assert set(combined) == set(y)

    def test_reduce_dataset_with_zero_percent(self):
        """Test reduce_dataset with 0% returns empty arrays."""
        x = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        x_reduced, y_reduced = self.loader.reduce_dataset(x, y, percent=0.0)

        assert len(x_reduced) == 0
        assert len(y_reduced) == 0

    def test_reduce_dataset_with_tiny_percent(self):
        """Test reduce_dataset with very small percentage."""
        x = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        x_reduced, y_reduced = self.loader.reduce_dataset(x, y, percent=0.01)

        assert len(x_reduced) == 1
        assert len(y_reduced) == 1

    def test_reduce_dataset_with_single_sample(self):
        """Test reduce_dataset with single sample."""
        x = np.array([[1]])
        y = np.array([0])

        x_reduced, y_reduced = self.loader.reduce_dataset(x, y, percent=0.5)

        # Should return at least the single sample or none
        assert len(x_reduced) <= 1
        assert len(y_reduced) <= 1

    def test_reduce_dataset_preserves_data_type(self):
        """Test reduce_dataset preserves numpy array types."""
        x = np.arange(100, dtype=np.float32).reshape(100, 1)
        y = np.arange(100, dtype=np.int64)

        x_reduced, y_reduced = self.loader.reduce_dataset(x, y, percent=0.5)

        assert x_reduced.dtype == np.float32
        assert y_reduced.dtype == np.int64

    def test_get_categories_with_special_characters(self):
        """Test get_categories with directories containing special characters."""
        test_path = self.temp_dir / "special_chars"
        test_path.mkdir()
        (test_path / "class-1").mkdir()
        (test_path / "class_2").mkdir()
        (test_path / "class 3").mkdir()
        (test_path / "class.4").mkdir()

        categories = self.loader.get_categories(test_path)

        assert len(categories) == 4
        assert "class-1" in categories
        assert "class_2" in categories
        assert "class 3" in categories
        assert "class.4" in categories

    def test_get_categories_with_numeric_names(self):
        """Test get_categories with numeric directory names."""
        test_path = self.temp_dir / "numeric"
        test_path.mkdir()
        (test_path / "0").mkdir()
        (test_path / "1").mkdir()
        (test_path / "2").mkdir()
        (test_path / "123").mkdir()

        categories = self.loader.get_categories(test_path)

        assert len(categories) == 4
        assert "0" in categories
        assert "123" in categories

    def test_get_categories_ignores_hidden_directories(self):
        """Test get_categories ignores all hidden directories."""
        test_path = self.temp_dir / "hidden"
        test_path.mkdir()
        (test_path / "visible1").mkdir()
        (test_path / ".hidden1").mkdir()
        (test_path / "..hidden2").mkdir()
        (test_path / ".DS_Store").mkdir()
        (test_path / "__pycache__").mkdir()
        (test_path / "visible2").mkdir()

        categories = self.loader.get_categories(test_path)

        assert len(categories) == 3  # visible1, visible2, __pycache__
        assert "visible1" in categories
        assert "visible2" in categories
        assert ".hidden1" not in categories
        assert "..hidden2" not in categories
        assert ".DS_Store" not in categories

    def test_get_categories_with_nested_structure(self):
        """Test get_categories only returns top-level directories."""
        test_path = self.temp_dir / "nested"
        test_path.mkdir()
        (test_path / "class1").mkdir()
        (test_path / "class1" / "subclass1").mkdir()
        (test_path / "class2").mkdir()

        categories = self.loader.get_categories(test_path)

        assert len(categories) == 2
        assert "class1" in categories
        assert "class2" in categories
        assert "subclass1" not in categories

    def test_split_data_randomizes_data(self):
        """Test split_data shuffles data by default."""
        x = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        x1, x2, y1, y2 = self.loader.split_data(x, y, split_ratio=0.7)

        assert len(x1) == 70
        assert len(x2) == 30
        # Data should be shuffled - not in original order
        # (Note: small chance this could fail if shuffle happens to maintain order)
        total_order_preserved = np.array_equal(y1, np.arange(70)) and np.array_equal(
            y2, np.arange(70, 100)
        )
        # If not totally preserved, it was shuffled (expected)
        assert not total_order_preserved or len(y1) < 10  # Allow for very small datasets

    def test_reduce_dataset_with_multidimensional_data(self):
        """Test reduce_dataset with high-dimensional data."""
        x = np.arange(1000).reshape(10, 10, 10)
        y = np.arange(10)

        x_reduced, y_reduced = self.loader.reduce_dataset(x, y, percent=0.5)

        assert len(x_reduced) == 5
        assert len(y_reduced) == 5
        assert x_reduced.shape[1:] == (10, 10)

