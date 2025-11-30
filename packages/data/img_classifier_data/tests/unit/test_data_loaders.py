"""Unit tests for data loading modules using mocking.

These tests verify data loader behavior with mocked dependencies
to ensure fast execution and isolation from other packages.
"""

import numpy as np
import pytest

from img_classifier_data import BaseDataLoader, ImageDataLoader


@pytest.mark.unit
class TestBaseDataLoader:
    """Unit tests for BaseDataLoader abstract class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir, mock_config):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = mock_config

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
        self.config.train_path.mkdir(parents=True, exist_ok=True)
        self.config.test_path.mkdir(parents=True, exist_ok=True)
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


@pytest.mark.unit
class TestImageDataLoaderCacheOperations:
    """Tests for ImageDataLoader cache operations."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir, mock_config):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = mock_config
        self.loader = ImageDataLoader(self.config)
        yield

    def test_save_and_load_cache_train(self):
        """Test saving and loading training data cache."""
        x = np.random.rand(10, 128, 128, 3).astype("float32")
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])

        # Save to cache
        self.loader._save_to_cache(x, y, train=True)

        # Load from cache
        cached = self.loader._load_from_cache(train=True)

        assert cached is not None
        x_cached, y_cached = cached
        np.testing.assert_array_equal(x, x_cached)
        np.testing.assert_array_equal(y, y_cached)

    def test_save_and_load_cache_test(self):
        """Test saving and loading test data cache."""
        x = np.random.rand(5, 128, 128, 3).astype("float32")
        y = np.array([0, 1, 2, 3, 0])

        # Save to cache
        self.loader._save_to_cache(x, y, train=False)

        # Load from cache
        cached = self.loader._load_from_cache(train=False)

        assert cached is not None
        x_cached, y_cached = cached
        np.testing.assert_array_equal(x, x_cached)
        np.testing.assert_array_equal(y, y_cached)

    def test_load_from_corrupted_cache(self):
        """Test loading from corrupted cache returns None."""
        # Create corrupted cache files
        x_path, y_path = self.loader._get_cache_path(train=True)
        x_path.parent.mkdir(parents=True, exist_ok=True)
        x_path.write_text("corrupted data")
        y_path.write_text("corrupted data")

        result = self.loader._load_from_cache(train=True)
        assert result is None

    def test_cache_path_creates_parent_directory(self):
        """Test that save_to_cache creates parent directories."""
        # Ensure cache dir doesn't exist
        import shutil

        if self.config.cache_dir.exists():
            shutil.rmtree(self.config.cache_dir)

        x = np.random.rand(5, 128, 128, 3).astype("float32")
        y = np.array([0, 1, 2, 3, 0])

        self.loader._save_to_cache(x, y, train=True)

        assert self.config.cache_dir.exists()

    def test_load_cache_with_only_x_file(self):
        """Test loading cache when only X file exists."""
        x_path, y_path = self.loader._get_cache_path(train=True)
        x_path.parent.mkdir(parents=True, exist_ok=True)

        import pickle

        with open(x_path, "wb") as f:  # type: ignore[arg-type]
            pickle.dump(np.array([1, 2, 3]), f)  # type: ignore[arg-type]

        result = self.loader._load_from_cache(train=True)
        assert result is None

    def test_load_cache_with_only_y_file(self):
        """Test loading cache when only y file exists."""
        x_path, y_path = self.loader._get_cache_path(train=True)
        y_path.parent.mkdir(parents=True, exist_ok=True)

        import pickle

        with open(y_path, "wb") as f:  # type: ignore[arg-type]
            pickle.dump(np.array([0, 1, 2]), f)  # type: ignore[arg-type]

        result = self.loader._load_from_cache(train=True)
        assert result is None


@pytest.mark.unit
class TestImageDataLoaderDownloadPrepare:
    """Tests for ImageDataLoader download and prepare operations."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir, mock_config):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = mock_config
        self.loader = ImageDataLoader(self.config)
        yield

    def test_download_dataset_already_exists(self):
        """Test download_dataset when zip already exists."""
        self.config.dataset_zip_id = "test_id_123"
        zip_path = self.config.data_path / f"{self.config.dataset_name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_text("fake zip content")

        result = self.loader.download_dataset()
        assert result is True

    def test_prepare_dataset_missing_zip(self):
        """Test prepare_dataset fails when zip doesn't exist."""
        result = self.loader.prepare_dataset()
        assert result is False

    def test_prepare_dataset_with_empty_train_test(self):
        """Test prepare_dataset when train/test exist but are empty."""
        self.config.train_path.mkdir(parents=True, exist_ok=True)
        self.config.test_path.mkdir(parents=True, exist_ok=True)

        # Create zip file
        zip_path = self.config.data_path / f"{self.config.dataset_name}.zip"
        zip_path.write_text("fake zip")

        result = self.loader.prepare_dataset()
        # Will fail because extraction will fail with fake zip
        assert result is False

    def test_setup_with_valid_structure(self):
        """Test setup with valid directory structure."""
        # Create valid train/test structure
        self.config.train_path.mkdir(parents=True, exist_ok=True)
        self.config.test_path.mkdir(parents=True, exist_ok=True)
        (self.config.train_path / "class1").mkdir()
        (self.config.train_path / "class2").mkdir()
        (self.config.test_path / "class1").mkdir()
        (self.config.test_path / "class2").mkdir()

        result = self.loader.setup()
        assert result is True

    def test_setup_with_no_categories(self):
        """Test setup fails when no categories found."""
        self.config.train_path.mkdir(parents=True, exist_ok=True)
        self.config.test_path.mkdir(parents=True, exist_ok=True)

        result = self.loader.setup()
        assert result is False

    def test_setup_missing_test_path(self):
        """Test setup fails when test path missing."""
        self.config.train_path.mkdir(parents=True, exist_ok=True)
        (self.config.train_path / "class1").mkdir()

        result = self.loader.setup()
        assert result is False

    def test_setup_missing_train_path(self):
        """Test setup fails when train path missing."""
        self.config.test_path.mkdir(parents=True, exist_ok=True)
        (self.config.test_path / "class1").mkdir()

        result = self.loader.setup()
        assert result is False


@pytest.mark.unit
class TestImageDataLoaderProcessing:
    """Tests for ImageDataLoader image processing."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir, mock_config):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = mock_config
        self.loader = ImageDataLoader(self.config)
        yield

    def test_process_images_no_categories(self):
        """Test _process_images raises error when no categories found."""
        test_path = self.temp_dir / "empty"
        test_path.mkdir()

        with pytest.raises(ValueError, match="No categories found"):
            self.loader._process_images(test_path)

    def test_process_images_with_valid_images(self, mocker):
        """Test _process_images successfully processes valid images."""
        # Create test directory structure
        test_path = self.temp_dir / "images"
        test_path.mkdir()
        (test_path / "class1").mkdir()
        (test_path / "class2").mkdir()

        # Create mock image files
        img1 = test_path / "class1" / "img1.jpg"
        img2 = test_path / "class1" / "img2.png"
        img3 = test_path / "class2" / "img3.jpeg"

        for img_file in [img1, img2, img3]:
            img_file.write_text("fake image data")

        # Mock cv2 functions - use config's image_size
        img_h, img_w = self.config.image_size
        mock_image = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
        mocker.patch("cv2.imread", return_value=mock_image)
        mocker.patch("cv2.resize", return_value=mock_image)

        # Mock tqdm to avoid progress bars in tests
        mocker.patch("img_classifier_data.image_loader.tqdm", side_effect=lambda x, **kwargs: x)

        x, y = self.loader._process_images(test_path)

        assert x.shape[0] == 3  # 3 images
        assert y.shape[0] == 3
        assert x.dtype == np.float32
        assert np.all(x >= 0) and np.all(x <= 1)  # Normalized
        assert self.loader.num_categories == 2

    def test_process_images_with_none_image(self, mocker):
        """Test _process_images skips images that fail to load."""
        test_path = self.temp_dir / "images"
        test_path.mkdir()
        (test_path / "class1").mkdir()

        # Create mock image files
        img1 = test_path / "class1" / "img1.jpg"
        img2 = test_path / "class1" / "img2.jpg"
        img1.write_text("fake")
        img2.write_text("fake")

        # Mock cv2.imread to return None for first image, valid for second
        img_h, img_w = self.config.image_size
        mock_image = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
        mocker.patch("cv2.imread", side_effect=[None, mock_image])
        mocker.patch("cv2.resize", return_value=mock_image)
        mocker.patch("img_classifier_data.image_loader.tqdm", side_effect=lambda x, **kwargs: x)

        x, y = self.loader._process_images(test_path)

        assert x.shape[0] == 1  # Only 1 image successfully processed

    def test_process_images_with_exception(self, mocker):
        """Test _process_images handles exceptions during processing."""
        test_path = self.temp_dir / "images"
        test_path.mkdir()
        (test_path / "class1").mkdir()

        img1 = test_path / "class1" / "img1.jpg"
        img2 = test_path / "class1" / "img2.jpg"
        img1.write_text("fake")
        img2.write_text("fake")

        img_h, img_w = self.config.image_size
        mock_image = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
        # First image raises exception, second succeeds
        mocker.patch("cv2.imread", side_effect=[Exception("Read error"), mock_image])
        mocker.patch("cv2.resize", return_value=mock_image)
        mocker.patch("img_classifier_data.image_loader.tqdm", side_effect=lambda x, **kwargs: x)

        x, y = self.loader._process_images(test_path)

        assert x.shape[0] == 1  # Only 1 image successfully processed

    def test_process_images_ignores_non_image_files(self, mocker):
        """Test _process_images ignores non-image files."""
        test_path = self.temp_dir / "images"
        test_path.mkdir()
        (test_path / "class1").mkdir()

        # Create various file types
        (test_path / "class1" / "img1.jpg").write_text("fake")
        (test_path / "class1" / "doc.txt").write_text("text")
        (test_path / "class1" / "data.csv").write_text("csv")
        (test_path / "class1" / "img2.png").write_text("fake")

        img_h, img_w = self.config.image_size
        mock_image = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
        mocker.patch("cv2.imread", return_value=mock_image)
        mocker.patch("cv2.resize", return_value=mock_image)
        mocker.patch("img_classifier_data.image_loader.tqdm", side_effect=lambda x, **kwargs: x)

        x, y = self.loader._process_images(test_path)

        assert x.shape[0] == 2  # Only 2 image files

    def test_process_images_sets_categories(self, mocker):
        """Test _process_images properly sets categories and num_categories."""
        test_path = self.temp_dir / "images"
        test_path.mkdir()
        (test_path / "cat").mkdir()
        (test_path / "dog").mkdir()
        (test_path / "bird").mkdir()

        (test_path / "cat" / "img.jpg").write_text("fake")
        (test_path / "dog" / "img.jpg").write_text("fake")
        (test_path / "bird" / "img.jpg").write_text("fake")

        img_h, img_w = self.config.image_size
        mock_image = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
        mocker.patch("cv2.imread", return_value=mock_image)
        mocker.patch("cv2.resize", return_value=mock_image)
        mocker.patch("img_classifier_data.image_loader.tqdm", side_effect=lambda x, **kwargs: x)

        x, y = self.loader._process_images(test_path)

        assert self.loader.num_categories == 3
        assert len(self.loader.categories) == 3
        assert set(y) == {0, 1, 2}  # Labels should be 0, 1, 2


@pytest.mark.unit
class TestImageDataLoaderLoadMethods:
    """Tests for load_train_data and load_test_data methods."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir, mock_config):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = mock_config
        self.loader = ImageDataLoader(self.config)
        yield

    def test_load_train_data_from_cache(self, mocker):
        """Test load_train_data loads from cache when available."""
        # Mock _load_from_cache to return data
        mock_x = np.random.rand(10, 128, 128, 3).astype("float32")
        mock_y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        mocker.patch.object(self.loader, "_load_from_cache", return_value=(mock_x, mock_y))

        X, y = self.loader.load_train_data()

        np.testing.assert_array_equal(X, mock_x)
        np.testing.assert_array_equal(y, mock_y)

    def test_load_train_data_processes_images(self, mocker):
        """Test load_train_data processes images when cache not available."""
        # Create directory structure
        self.config.train_path.mkdir(parents=True, exist_ok=True)
        (self.config.train_path / "class1").mkdir()
        (self.config.train_path / "class1" / "img.jpg").write_text("fake")

        # Mock methods
        mock_x = np.random.rand(5, 128, 128, 3).astype("float32")
        mock_y = np.array([0, 1, 0, 1, 0])

        mocker.patch.object(self.loader, "_load_from_cache", return_value=None)
        mocker.patch.object(self.loader, "_process_images", return_value=(mock_x, mock_y))
        mock_save = mocker.patch.object(self.loader, "_save_to_cache")

        x, y = self.loader.load_train_data()

        np.testing.assert_array_equal(x, mock_x)
        np.testing.assert_array_equal(y, mock_y)
        mock_save.assert_called_once_with(mock_x, mock_y, train=True)

    def test_load_test_data_from_cache(self, mocker):
        """Test load_test_data loads from cache when available."""
        mock_x = np.random.rand(5, 128, 128, 3).astype("float32")
        mock_y = np.array([0, 1, 2, 0, 1])
        mocker.patch.object(self.loader, "_load_from_cache", return_value=(mock_x, mock_y))

        x, y = self.loader.load_test_data()

        np.testing.assert_array_equal(x, mock_x)
        np.testing.assert_array_equal(y, mock_y)

    def test_load_test_data_processes_images(self, mocker):
        """Test load_test_data processes images when cache not available."""
        self.config.test_path.mkdir(parents=True, exist_ok=True)
        (self.config.test_path / "class1").mkdir()
        (self.config.test_path / "class1" / "img.jpg").write_text("fake")

        mock_x = np.random.rand(3, 128, 128, 3).astype("float32")
        mock_y = np.array([0, 1, 2])

        mocker.patch.object(self.loader, "_load_from_cache", return_value=None)
        mocker.patch.object(self.loader, "_process_images", return_value=(mock_x, mock_y))
        mock_save = mocker.patch.object(self.loader, "_save_to_cache")

        x, y = self.loader.load_test_data()

        np.testing.assert_array_equal(x, mock_x)
        np.testing.assert_array_equal(y, mock_y)
        mock_save.assert_called_once_with(mock_x, mock_y, train=False)


@pytest.mark.unit
class TestImageDataLoaderCacheErrors:
    """Tests for cache error handling."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir, mock_config):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = mock_config
        self.loader = ImageDataLoader(self.config)
        yield

    def test_save_to_cache_handles_write_errors(self, mocker):
        """Test _save_to_cache handles write errors gracefully."""
        x = np.random.rand(5, 128, 128, 3).astype("float32")
        y = np.array([0, 1, 2, 0, 1])

        # Mock pickle.dump to raise an exception
        mocker.patch("pickle.dump", side_effect=Exception("Write error"))

        # Should not raise, just print error
        self.loader._save_to_cache(x, y, train=True)

        # Verify cache files were not created or are incomplete
        x_path, y_path = self.loader._get_cache_path(train=True)
        # Files might exist but be incomplete/corrupted

    def test_save_to_cache_creates_directory(self):
        """Test _save_to_cache creates cache directory if it doesn't exist."""
        import shutil

        if self.config.cache_dir.exists():
            shutil.rmtree(self.config.cache_dir)

        x = np.random.rand(2, 128, 128, 3).astype("float32")
        y = np.array([0, 1])

        self.loader._save_to_cache(x, y, train=True)

        assert self.config.cache_dir.exists()


@pytest.mark.unit
class TestImageDataLoaderDownloadSuccess:
    """Tests for successful download scenarios."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir, mock_config):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = mock_config
        self.loader = ImageDataLoader(self.config)
        yield

    def test_download_dataset_success(self, mocker):
        """Test successful download from Google Drive."""
        self.config.dataset_zip_id = "test_id_123"

        # Mock download_from_google_drive to return True
        mocker.patch(
            "img_classifier_data.image_loader.download_from_google_drive", return_value=True
        )

        result = self.loader.download_dataset()

        assert result is True

    def test_download_dataset_failure(self, mocker):
        """Test failed download from Google Drive."""
        self.config.dataset_zip_id = "test_id_123"

        # Mock download_from_google_drive to return False
        mocker.patch(
            "img_classifier_data.image_loader.download_from_google_drive", return_value=False
        )

        result = self.loader.download_dataset()

        assert result is False


@pytest.mark.unit
class TestImageDataLoaderPrepareExtraction:
    """Tests for dataset extraction scenarios."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir, mock_config):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = mock_config
        self.loader = ImageDataLoader(self.config)
        yield

    def test_prepare_dataset_extracts_archive(self, mocker):
        """Test prepare_dataset extracts archive when needed."""
        # Create zip file
        zip_path = self.config.data_path / f"{self.config.dataset_name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_text("fake zip content")

        # Mock extract_archive and organize_dataset
        mocker.patch("img_classifier_data.image_loader.extract_archive", return_value=True)
        mocker.patch("img_classifier_data.image_loader.organize_dataset")

        # Create extracted path after "extraction"
        extracted_path = self.config.data_path / self.config.dataset_name
        extracted_path.mkdir(parents=True, exist_ok=True)

        result = self.loader.prepare_dataset()

        assert result is True

    def test_prepare_dataset_extraction_fails(self, mocker):
        """Test prepare_dataset returns False when extraction fails."""
        zip_path = self.config.data_path / f"{self.config.dataset_name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_text("fake zip content")

        # Mock extract_archive to return False
        mocker.patch("img_classifier_data.image_loader.extract_archive", return_value=False)

        result = self.loader.prepare_dataset()

        assert result is False

    def test_prepare_dataset_organizes_when_extracted(self, mocker):
        """Test prepare_dataset organizes dataset when already extracted."""
        # Create extracted directory
        extracted_path = self.config.data_path / self.config.dataset_name
        extracted_path.mkdir(parents=True, exist_ok=True)

        # Mock organize_dataset
        mock_organize = mocker.patch("img_classifier_data.image_loader.organize_dataset")

        result = self.loader.prepare_dataset()

        # organize_dataset should be called
        mock_organize.assert_called_once_with(extracted_path, self.config.data_path)
        assert result is True
