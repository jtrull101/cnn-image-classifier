"""Image data loader implementation."""

import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
from img_classifier_config import BaseConfig, DatasetDetector
from img_classifier_utils import (
    download_from_google_drive,
    ensure_directory_exists,
    extract_archive,
    organize_dataset,
)
from tqdm import tqdm

from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)

_PIXEL_MAX = 255.0


class ImageDataLoader(BaseDataLoader):
    """Data loader for image classification tasks.

    This loader handles:
    - Downloading datasets from Google Drive
    - Extracting and organizing image files
    - Loading and preprocessing images
    - Caching processed data for faster reloading
    """

    def __init__(self, config: BaseConfig):
        """Initialize the image data loader.

        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.config.create_directories()

    def download_dataset(self) -> bool:
        """Download dataset from Google Drive if not present.

        Returns:
            True if successful or already present, False otherwise
        """
        if self.config.data_path is None:
            raise RuntimeError("data_path should be set")

        if self.config.dataset_zip_id is None:
            logger.info("No dataset zip ID provided, skipping download")
            return False

        zip_path = self.config.data_path / f"{self.config.dataset_name}.zip"

        if zip_path.exists():
            logger.info("Dataset already downloaded at %s", zip_path)
            return True

        logger.info("Downloading dataset from Google Drive...")
        return download_from_google_drive(self.config.dataset_zip_id, zip_path, quiet=False)

    def prepare_dataset(self) -> bool:
        """Extract and organize the dataset.

        Returns:
            True if successful, False otherwise
        """
        if self.config.data_path is None:
            raise RuntimeError("data_path should be set")
        if self.config.train_path is None:
            raise RuntimeError("train_path should be set")
        if self.config.test_path is None:
            raise RuntimeError("test_path should be set")

        zip_path = self.config.data_path / f"{self.config.dataset_name}.zip"

        # Check if already prepared
        if self.config.train_path.exists() and self.config.test_path.exists():
            train_items = list(self.config.train_path.iterdir())
            test_items = list(self.config.test_path.iterdir())
            if train_items and test_items:
                logger.info("Dataset already prepared")
                return True

        # Extract if not already extracted
        extracted_path = self.config.data_path / self.config.dataset_name
        if not extracted_path.exists():
            if not zip_path.exists():
                logger.warning("Dataset zip not found at %s", zip_path)
                return False

            logger.info("Extracting dataset...")
            if not extract_archive(zip_path, self.config.data_path):
                return False

        # Organize the dataset structure if needed
        if extracted_path.exists():
            logger.info("Organizing dataset...")
            organize_dataset(extracted_path, self.config.data_path)

        return True

    def _get_cache_path(self, train: bool) -> tuple[Path, Path]:
        """Get cache file paths for X and y data.

        Args:
            train: Whether this is training data

        Returns:
            Tuple of (X_cache_path, y_cache_path)
        """
        split = "train" if train else "test"
        x_path = self.config.cache_dir / f"X_data_{split}.pkl"
        y_path = self.config.cache_dir / f"y_data_{split}.pkl"
        return x_path, y_path

    def _load_from_cache(self, train: bool) -> tuple[np.ndarray, np.ndarray] | None:
        """Load data from cache if available.

        Args:
            train: Whether this is training data

        Returns:
            Tuple of (X, y) if cache exists, None otherwise
        """
        x_path, y_path = self._get_cache_path(train)

        if x_path.exists() and y_path.exists():
            try:
                logger.info("Loading %s data from cache...", "train" if train else "test")
                with open(x_path, "rb") as f:
                    X = pickle.load(f)
                with open(y_path, "rb") as f:
                    y = pickle.load(f)
                return X, y
            except Exception:
                logger.exception("Error loading from cache")
                return None

        return None

    def _save_to_cache(self, X: np.ndarray, y: np.ndarray, train: bool) -> None:
        """Save data to cache.

        Args:
            X: Feature array
            y: Label array
            train: Whether this is training data
        """
        ensure_directory_exists(self.config.cache_dir)
        x_path, y_path = self._get_cache_path(train)

        try:
            with open(x_path, "wb") as f:
                pickle.dump(X, f)
            with open(y_path, "wb") as f:
                pickle.dump(y, f)
            logger.info("Data cached successfully")
        except Exception:
            logger.exception("Error saving to cache")

    def _process_images(self, data_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Process images from a directory.

        Args:
            data_path: Path to directory containing category subdirectories

        Returns:
            Tuple of (X, y) numpy arrays
        """
        categories = self.get_categories(data_path)
        if not categories:
            raise ValueError(f"No categories found in {data_path}")

        logger.info("Found categories: %s", categories)
        self.categories = categories
        self.num_categories = len(categories)

        image_data = []

        for category_idx, category in enumerate(tqdm(categories, desc="Processing categories")):
            category_path = data_path / category

            image_files = [
                f
                for f in category_path.iterdir()
                if f.suffix.lower() in DatasetDetector.SUPPORTED_IMAGE_EXTENSIONS
            ]

            for img_path in tqdm(image_files, desc=f"Processing {category}", leave=False):
                try:
                    # Read and resize image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    image_resized = cv2.resize(image, self.config.image_size)
                    image_data.append([image_resized, category_idx])

                except Exception:
                    logger.exception("Error processing %s", img_path)

        # Convert to numpy arrays
        data_array = np.array(image_data, dtype=object)

        X = np.array([x[0] for x in data_array])
        y = np.array([x[1] for x in data_array])

        # Normalize pixel values to [0, 1]
        X = X.astype("float32") / _PIXEL_MAX

        # Reshape to proper format
        X = X.reshape(-1, *self.config.image_size, self.config.color_channels)

        return X, y

    def load_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load training data.

        Returns:
            Tuple of (X_train, y_train)
        """
        if self.config.train_path is None:
            raise RuntimeError("train_path should be set")

        # Try loading from cache first
        cached = self._load_from_cache(train=True)
        if cached is not None:
            return cached

        # Process images from disk
        X, y = self._process_images(self.config.train_path)

        # Cache the processed data
        self._save_to_cache(X, y, train=True)

        return X, y

    def load_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load test data.

        Returns:
            Tuple of (X_test, y_test)
        """
        if self.config.test_path is None:
            raise RuntimeError("test_path should be set")

        # Try loading from cache first
        cached = self._load_from_cache(train=False)
        if cached is not None:
            return cached

        # Process images from disk
        X, y = self._process_images(self.config.test_path)

        # Cache the processed data
        self._save_to_cache(X, y, train=False)

        return X, y

    def setup(self) -> bool:
        """Complete setup: download, prepare, and validate dataset.

        Returns:
            True if successful, False otherwise
        """
        if self.config.train_path is None:
            raise RuntimeError("train_path should be set")
        if self.config.test_path is None:
            raise RuntimeError("test_path should be set")

        # Download dataset
        self.download_dataset()

        # Prepare (extract and organize)
        if not self.prepare_dataset():
            return False

        # Validate that we have data
        if not self.config.train_path.exists() or not self.config.test_path.exists():
            logger.warning("Dataset not properly prepared")
            return False

        train_categories = self.get_categories(self.config.train_path)
        test_categories = self.get_categories(self.config.test_path)

        if not train_categories or not test_categories:
            logger.warning("No categories found in dataset")
            return False

        logger.info("Dataset setup complete. Found %d categories.", len(train_categories))
        return True
