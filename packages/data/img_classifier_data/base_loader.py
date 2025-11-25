"""Base data loader interface."""

from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
from pathlib import Path

from img_classifier_config import BaseConfig


class BaseDataLoader(ABC):
    """Abstract base class for data loaders.

    This class defines the interface that all data loaders must implement,
    making it easy to create new loaders for different datasets.
    """

    def __init__(self, config: BaseConfig):
        """Initialize the data loader.

        Args:
            config: Configuration object containing data settings
        """
        self.config = config
        self.categories: List[str] = []
        self.num_categories: int = 0

    @abstractmethod
    def load_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data.

        Returns:
            Tuple of (X_train, y_train) numpy arrays
        """
        pass

    @abstractmethod
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data.

        Returns:
            Tuple of (X_test, y_test) numpy arrays
        """
        pass

    @abstractmethod
    def download_dataset(self) -> bool:
        """Download the dataset if not present.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def prepare_dataset(self) -> bool:
        """Prepare the dataset (extract, organize, etc.).

        Returns:
            True if successful, False otherwise
        """
        pass

    def get_categories(self, data_path: Path) -> List[str]:
        """Get list of categories from a directory.

        Args:
            data_path: Path to directory containing category subdirectories

        Returns:
            List of category names
        """
        if not data_path.exists():
            return []

        categories = [
            d.name for d in data_path.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        return sorted(categories)

    def split_data(
        self, X: np.ndarray, y: np.ndarray, split_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into two sets.

        Args:
            X: Feature array
            y: Label array
            split_ratio: Ratio for first split

        Returns:
            Tuple of (X1, X2, y1, y2)
        """
        split_idx = int(len(X) * split_ratio)
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        split_indices = indices[:split_idx]
        rest_indices = indices[split_idx:]

        return (X[split_indices], X[rest_indices], y[split_indices], y[rest_indices])

    def reduce_dataset(
        self, X: np.ndarray, y: np.ndarray, percent: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce dataset to a percentage of original size.

        Args:
            X: Feature array
            y: Label array
            percent: Percentage to keep (0.0 to 1.0)

        Returns:
            Tuple of (X_reduced, y_reduced)
        """
        if percent >= 1.0:
            return X, y

        indices = np.arange(int(percent * len(X)))
        np.random.shuffle(indices)

        return X[indices], y[indices]
