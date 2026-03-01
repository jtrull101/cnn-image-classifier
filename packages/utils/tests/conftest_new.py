"""Root conftest.py for shared pytest configuration and fixtures."""

import os
import shutil
from collections.abc import Generator
from pathlib import Path

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
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "requires_data: marks tests that require dataset download")

    # Set environment variables for testing
    os.environ["TESTING"] = "1"
    os.environ["PYTEST_XDIST_WORKER"] = config.getoption("dist", "no")


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


@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path]:
    """Provide a temporary directory for test data that persists across the session."""
    base_dir = Path(__file__).parent / ".test_data"
    base_dir.mkdir(exist_ok=True)
    yield base_dir
    # Cleanup after all tests
    if base_dir.exists():
        shutil.rmtree(base_dir, ignore_errors=True)


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
