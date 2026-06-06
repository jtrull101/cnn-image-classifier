"""Base data loader interface."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

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
        self.categories: list[str] = []
        self.num_categories: int = 0

    @abstractmethod
    def load_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load training data.

        Returns:
            Tuple of (X_train, y_train) numpy arrays
        """
        pass

    @abstractmethod
    def load_test_data(self) -> tuple[np.ndarray, np.ndarray]:
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

    def get_categories(self, data_path: Path) -> list[str]:
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
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_ratio: float = 0.5,
        seed: int | None = None,
        stratify: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into two sets.

        Args:
            X: Feature array
            y: Label array (1-D labels or one-hot)
            split_ratio: Ratio for first split
            seed: If set, shuffle deterministically via a local RNG that does not disturb the
                global NumPy state (so repeated calls yield the same partition).
            stratify: If True, split each class by ``split_ratio`` so both halves keep the
                class balance — important when downstream code selects models by accuracy.

        Returns:
            Tuple of (X1, X2, y1, y2)
        """
        rng = np.random.default_rng(seed) if seed is not None else None

        def _shuffle(arr: np.ndarray) -> None:
            (rng.shuffle if rng is not None else np.random.shuffle)(arr)

        if stratify and len(X) > 0:
            labels = np.argmax(y, axis=1) if y.ndim > 1 else y
            first_parts, rest_parts = [], []
            for cls in np.unique(labels):
                cls_indices = np.where(labels == cls)[0]
                _shuffle(cls_indices)
                k = int(len(cls_indices) * split_ratio)
                first_parts.append(cls_indices[:k])
                rest_parts.append(cls_indices[k:])
            split_indices = np.concatenate(first_parts) if first_parts else np.array([], dtype=int)
            rest_indices = np.concatenate(rest_parts) if rest_parts else np.array([], dtype=int)
            # Re-shuffle the merged halves so classes aren't grouped together.
            _shuffle(split_indices)
            _shuffle(rest_indices)
        else:
            indices = np.arange(len(X))
            _shuffle(indices)
            split_idx = int(len(X) * split_ratio)
            split_indices = indices[:split_idx]
            rest_indices = indices[split_idx:]

        return (X[split_indices], X[rest_indices], y[split_indices], y[rest_indices])

    def reduce_dataset(
        self, X: np.ndarray, y: np.ndarray, percent: float
    ) -> tuple[np.ndarray, np.ndarray]:
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

        total = len(X)
        keep = max(int(percent * total), 0)
        if keep == 0:
            return X[:0], y[:0]

        indices = np.arange(total)
        np.random.shuffle(indices)
        selected = indices[:keep]

        return X[selected], y[selected]
