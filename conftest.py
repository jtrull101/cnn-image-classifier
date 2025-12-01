"""Root conftest.py for shared pytest configuration and fixtures."""

import os
import shutil
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock

import pytest


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "serial: marks tests that must run serially (not in parallel)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slow, use real dependencies)"
    )
    config.addinivalue_line("markers", "unit: marks tests as unit tests (fast, use mocking)")
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests (critical path validation)"
    )
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "requires_data: marks tests that require dataset download")

    # Set environment variables for testing
    os.environ["TESTING"] = "1"
    os.environ["PYTEST_XDIST_WORKER"] = config.getoption("dist", "no")


def pytest_sessionfinish(session, exitstatus):
    """Clean up after test session, handling Windows permission issues."""
    import warnings

    # Suppress Windows permission errors during cleanup
    # These are harmless and occur when pytest tries to clean up temp symlinks
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # On Windows, symlink cleanup can fail - ignore those errors
    if os.name == "nt":
        import atexit

        # Clear atexit handlers that might cause permission errors
        # This prevents the "PermissionError: [WinError 5]" during cleanup
        atexit._clear()


def pytest_collection_modifyitems(config, items):
    """Modify test items to add markers and groups for parallel execution."""
    for item in items:
        # Add 'unit' marker to all tests by default if not marked
        if not any(
            marker.name in ["unit", "integration", "slow"] for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)

        # Group tests by module for better parallel distribution
        if hasattr(item, "module"):
            module_name = item.module.__name__
            item.add_marker(pytest.mark.xdist_group(module_name))

        # Tests marked as 'serial' should run in a dedicated group
        if "serial" in [marker.name for marker in item.iter_markers()]:
            item.add_marker(pytest.mark.xdist_group("serial"))


# =============================================================================
# Shared Test Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path]:
    """Provide a temporary directory for test data that persists across the session."""
    temp_dir = Path("./test_data_temp")
    temp_dir.mkdir(exist_ok=True)
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def isolated_tmp_dir(tmp_path_factory, worker_id) -> Generator[Path]:
    """
    Provide an isolated temporary directory for each test, safe for parallel execution.

    This fixture uses pytest-xdist's worker_id to ensure each worker process
    gets its own isolated directory space.
    """
    if worker_id == "master":
        # Single process - use standard tmp_path
        temp_dir = tmp_path_factory.mktemp("test")
    else:
        # Multiple workers - create worker-specific directories
        root_tmp = tmp_path_factory.getbasetemp().parent
        temp_dir = root_tmp / f"worker_{worker_id}" / f"test_{os.getpid()}"
        temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Mock Fixtures for Unit Tests
# =============================================================================


@pytest.fixture
def mock_config(isolated_tmp_dir):
    """Provide a mocked configuration object for unit tests."""
    config = Mock()
    config.working_dir = isolated_tmp_dir
    config.data_path = isolated_tmp_dir / "data"
    config.train_path = isolated_tmp_dir / "data" / "train"
    config.test_path = isolated_tmp_dir / "data" / "test"
    config.dataset_name = "mock_dataset"
    config.dataset_zip_id = None
    config.models_dir = isolated_tmp_dir / "models"
    config.logs_dir = isolated_tmp_dir / "logs"
    config.cache_dir = isolated_tmp_dir / "data" / "cache"
    config.num_classes = 4
    config.class_names = ["class1", "class2", "class3", "class4"]
    config.input_shape = (128, 128, 3)
    config.image_size = (128, 128)
    config.color_channels = 3
    config.batch_size = 32
    config.num_epochs = 25
    config.learning_rate = 0.001
    config.validation_split = 0.2
    config.dropout_rate = 0.3
    config.data_percent = 1.0
    config.use_early_stopping = True
    config.use_model_checkpoint = True
    config.use_accuracy_threshold_stopping = True
    config.early_stopping_patience = 5
    config.accuracy_threshold = 0.995
    config.create_directories = Mock()

    def _create_dirs():
        paths = [
            config.working_dir,
            config.data_path,
            config.train_path,
            config.test_path,
            config.cache_dir,
            config.models_dir,
            config.logs_dir,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    config.create_directories.side_effect = _create_dirs
    config.create_directories()
    return config


@pytest.fixture
def mock_model(mock_config):
    """Provide a mocked model object for unit tests."""
    model = Mock()
    keras_model = Mock()
    keras_model.fit = Mock(return_value=Mock(history={"loss": [0.5], "val_loss": [0.6]}))
    keras_model.evaluate = Mock(return_value=[0.3, 0.92])
    keras_model.predict = Mock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    model.model = keras_model
    model.config = mock_config
    model.compile = Mock()
    model.save = Mock()
    model.load = Mock()
    return model


@pytest.fixture
def mock_data_loader(mock_config):
    """Provide a mocked data loader object for unit tests."""
    import numpy as np

    loader = Mock()
    loader.config = mock_config

    # Mock data loading methods
    x_train = np.random.rand(100, 128, 128, 3).astype("float32")
    y_train = np.random.randint(0, 4, 100)
    x_test = np.random.rand(40, 128, 128, 3).astype("float32")
    y_test = np.random.randint(0, 4, 40)

    loader.load_train_data = Mock(return_value=(x_train, y_train))
    loader.load_test_data = Mock(return_value=(x_test, y_test))
    loader.split_data = Mock(return_value=(x_test[:20], x_test[20:], y_test[:20], y_test[20:]))
    loader.reduce_dataset = Mock(side_effect=lambda x, y: (x[: len(x) // 2], y[: len(y) // 2]))
    loader.get_categories = Mock(return_value=["class1", "class2", "class3", "class4"])
    loader.download_dataset = Mock(return_value=True)
    loader.prepare_dataset = Mock(return_value=True)

    return loader


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment before any tests run."""
    # Create necessary directories
    log_dir = Path("/tmp/img_classifier_cnn/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set any environment variables needed for testing
    os.environ["IMG_CLASSIFIER_ENV"] = "test"

    yield

    # Cleanup after all tests
    # (Optional - you might want to keep logs)


@pytest.fixture(scope="function")
def mock_config_paths(isolated_tmp_dir):
    """Provide mock configuration paths for testing."""
    paths = {
        "working_dir": isolated_tmp_dir / "work",
        "data_path": isolated_tmp_dir / "data",
        "models_dir": isolated_tmp_dir / "models",
        "logs_dir": isolated_tmp_dir / "logs",
    }

    # Create directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


# Performance monitoring
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Track test execution time and add to report."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and hasattr(report, "duration") and report.duration > 5:
        # Flag slow tests
        report.longrepr = f"{report.longrepr}\n⚠️  SLOW TEST: {report.duration:.2f}s"
