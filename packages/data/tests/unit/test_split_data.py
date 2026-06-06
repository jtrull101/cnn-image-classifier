"""Tests for BaseDataLoader.split_data determinism and stratification.

These guard the eval-split reliability fix: a seeded split must be reproducible (so model
selection across attempts compares like-for-like), and a stratified split must preserve class
balance in both halves.
"""

import numpy as np
import pytest

from img_classifier_config import BaseConfig
from img_classifier_data.base_loader import BaseDataLoader


pytestmark = pytest.mark.unit


class _Loader(BaseDataLoader):
    """Minimal concrete loader so BaseDataLoader.split_data can be exercised."""

    def load_train_data(self):
        raise NotImplementedError

    def load_test_data(self):
        raise NotImplementedError

    def download_dataset(self):
        raise NotImplementedError

    def prepare_dataset(self):
        raise NotImplementedError


@pytest.fixture
def loader(tmp_path):
    return _Loader(BaseConfig(working_dir=tmp_path))


def _xy(n=100, num_classes=2, imbalance=None):
    X = np.arange(n).reshape(n, 1).astype("float32")
    if imbalance is not None:
        y = np.array(([0] * imbalance) + ([1] * (n - imbalance)))
    else:
        y = np.arange(n) % num_classes
    return X, y


def test_seeded_split_is_reproducible(loader):
    X, y = _xy()
    a = loader.split_data(X, y, 0.5, seed=1234)
    b = loader.split_data(X, y, 0.5, seed=1234)
    for left, right in zip(a, b, strict=True):
        assert np.array_equal(left, right)


def test_different_seeds_differ(loader):
    X, y = _xy()
    a_val = loader.split_data(X, y, 0.5, seed=1)[0]
    b_val = loader.split_data(X, y, 0.5, seed=2)[0]
    assert not np.array_equal(a_val, b_val)


def test_seeded_split_does_not_disturb_global_rng(loader):
    """A seeded split must not perturb global NumPy state used elsewhere for training seeds."""
    X, y = _xy()
    np.random.seed(0)
    expected = np.random.rand()
    np.random.seed(0)
    loader.split_data(X, y, 0.5, seed=1234)
    assert np.random.rand() == expected


def test_stratified_split_preserves_class_balance(loader):
    # 30 of class 0, 70 of class 1; a 50% stratified split should halve each class.
    X, y = _xy(n=100, imbalance=30)
    _X1, _X2, y1, y2 = loader.split_data(X, y, 0.5, seed=7, stratify=True)
    assert int((y1 == 0).sum()) == 15
    assert int((y1 == 1).sum()) == 35
    assert int((y2 == 0).sum()) == 15
    assert int((y2 == 1).sum()) == 35


def test_stratified_split_partitions_all_rows_without_overlap(loader):
    X, y = _xy(n=100, imbalance=30)
    X1, X2, _, _ = loader.split_data(X, y, 0.5, seed=7, stratify=True)
    combined = np.concatenate([X1.ravel(), X2.ravel()])
    assert sorted(combined.tolist()) == list(range(100))


def test_stratified_handles_one_hot_labels(loader):
    X, y_int = _xy(n=40, imbalance=10)
    y_onehot = np.eye(2)[y_int]
    _X1, _X2, y1, y2 = loader.split_data(X, y_onehot, 0.5, seed=3, stratify=True)
    # class 0 has 10 rows -> 5 each half; one-hot column 0 sums accordingly.
    assert int(y1[:, 0].sum()) == 5
    assert int(y2[:, 0].sum()) == 5


def test_unseeded_unstratified_still_works(loader):
    """Legacy path (no seed, no stratify) keeps the original behaviour/shape."""
    X, y = _xy(n=50)
    X1, X2, _y1, _y2 = loader.split_data(X, y, 0.4)
    assert len(X1) == 20
    assert len(X2) == 30
    assert len(X1) + len(X2) == 50
