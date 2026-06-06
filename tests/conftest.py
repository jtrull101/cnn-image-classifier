"""Test-specific conftest.py for the tests directory."""

import pytest


@pytest.fixture(scope="session")
def test_config_data(test_data_dir):
    """Provide test configuration data."""
    return {
        "project_name": "test_project",
        "num_classes": 4,
        "class_names": ["class1", "class2", "class3", "class4"],
        "batch_size": 32,
        "num_epochs": 5,
        "learning_rate": 0.001,
    }


@pytest.fixture
def temp_dataset_structure(isolated_tmp_dir):
    """Create a temporary dataset structure for testing."""
    dataset_dir = isolated_tmp_dir / "dataset"

    # Create train/test split
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    for split_dir in [train_dir, test_dir]:
        for class_name in ["class1", "class2", "class3", "class4"]:
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Create dummy image files
            for i in range(10):
                (class_dir / f"image_{i}.jpg").touch()

    return {
        "dataset_dir": dataset_dir,
        "train_dir": train_dir,
        "test_dir": test_dir,
    }


# Group markers for specific test modules
def pytest_collection_modifyitems(config, items):
    """Add specific markers for test modules."""
    for item in items:
        # Mark integration tests
        if "tests/integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.serial)  # Integration tests run serially

        # Mark slow tests
        if any(x in item.nodeid for x in ["test_training", "test_models"]):
            item.add_marker(pytest.mark.slow)

        # Group tests by their test file for better distribution
        if "test_config" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("config_tests"))
        elif "test_data_loaders" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("data_tests"))
        elif "test_models" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("model_tests"))
        elif "test_training" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("training_tests"))
        elif "test_utils" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("utils_tests"))
