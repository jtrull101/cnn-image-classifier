# Testing Guide

This document explains the testing structure and philosophy for the alz-mri-neural-network monorepo.

## Test Organization

The test suite is organized into two main categories:

### 1. Unit Tests (Fast, Mocked)

**Location**: `packages/*/tests/unit/` and `tests/unit/`

**Purpose**: Fast, isolated tests that verify individual components work correctly in isolation.

**Characteristics**:
- **Speed**: <1 second per test
- **Dependencies**: Use mocking for all external dependencies
- **I/O**: No file system or network operations
- **Isolation**: Test only the package's own code
- **Execution**: Run on every commit, part of CI fast path

**Example**:
```python
@pytest.mark.unit
def test_trainer_initialization(mock_config, mock_model, mock_data_loader):
    """Test that Trainer initializes with mocked dependencies."""
    trainer = Trainer(mock_config, mock_model, mock_data_loader)
    assert trainer.config == mock_config
```

### 2. Integration Tests (Slower, Real Dependencies)

**Location**: `tests/integration/`

**Purpose**: Verify that multiple packages work together correctly with real dependencies.

**Characteristics**:
- **Speed**: Variable (seconds to minutes)
- **Dependencies**: Use real implementations from multiple packages
- **I/O**: May perform actual file operations, data loading
- **Coverage**: Test end-to-end workflows and cross-package interactions
- **Execution**: Run on PR merges or manually, part of CI slow path

**Example**:
```python
@pytest.mark.integration
@pytest.mark.slow
def test_complete_training_workflow():
    """Test complete training workflow with real components."""
    config = DatasetConfig(...)
    loader = ImageDataLoader(config)
    model = ArchitectureFactory.create(config)
    trainer = Trainer(config, model, loader)
    results = trainer.run()
    assert results["accuracy"] > 0.5
```

## Test Markers

Use pytest markers to categorize tests:

| Marker | Description | Default Behavior |
|--------|-------------|------------------|
| `@pytest.mark.unit` | Fast unit tests with mocks | **Always run** |
| `@pytest.mark.integration` | Integration tests with real deps | Skipped by default |
| `@pytest.mark.slow` | Tests taking >10 seconds | Included with integration |
| `@pytest.mark.serial` | Must run serially, not parallel | Run in serial group |
| `@pytest.mark.requires_gpu` | Requires GPU access | Skip if no GPU |
| `@pytest.mark.requires_data` | Requires dataset download | Skip if no data |

## Running Tests

### Run All Unit Tests (Default)

```powershell
# Run all unit tests (fast, default behavior)
pytest

# Or explicitly
pytest -m unit
```

### Run All Integration Tests

```powershell
# Run only integration tests
pytest -m integration

# Run both unit and integration tests
pytest -m "unit or integration"
```

### Run Tests for Specific Package

```powershell
# Test only the training package
pytest packages/training/

# Test only config package
pytest packages/config/
```

### Run Tests by Speed

```powershell
# Run only fast tests (exclude slow)
pytest -m "unit and not slow"

# Run only slow tests
pytest -m slow
```

### Run Tests Serially (No Parallel)

```powershell
# Disable parallel execution
pytest -n 0

# Run only serial tests
pytest -m serial
```

## Package-Level Testing Rules

### Good Practices ✅

1. **Package unit tests should only test their own code**
   ```python
   # In packages/training/tests/unit/test_trainer.py
   from img_classifier_training import Trainer  # Own package
   from unittest.mock import Mock  # Mock external deps
   ```

2. **Use shared mock fixtures from root conftest.py**
   ```python
   def test_trainer(mock_config, mock_model, mock_data_loader):
       trainer = Trainer(mock_config, mock_model, mock_data_loader)
   ```

3. **Cross-package tests belong in tests/integration/**
   ```python
   # In tests/integration/test_training_workflow.py
   from img_classifier_config import DatasetConfig
   from img_classifier_data import ImageDataLoader
   from img_classifier_training import Trainer
   ```

### Bad Practices ❌

1. **Don't import other packages in package unit tests**
   ```python
   # BAD - in packages/training/tests/unit/
   from img_classifier_models import ArchitectureFactory  # Cross-package import
   ```

2. **Don't perform I/O in unit tests**
   ```python
   # BAD - in unit tests
   with open("real_file.txt") as f:  # Real file I/O
       data = f.read()
   ```

3. **Don't use real TensorFlow models in unit tests**
   ```python
   # BAD - in unit tests
   model = tf.keras.Sequential([...])  # Real TensorFlow
   model.fit(X, y)  # Actual training
   ```

## Shared Test Fixtures

The root `conftest.py` provides shared fixtures for all tests:

### For Unit Tests (Mocked)

- `mock_config`: Mocked configuration object
- `mock_model`: Mocked model with Keras interface
- `mock_data_loader`: Mocked data loader
- `isolated_tmp_dir`: Temporary directory (safe for parallel execution)

### For Integration Tests

- `test_data_dir`: Shared session-scoped temp directory
- `temp_dataset_structure`: Pre-created dataset directory structure

## Test Development Workflow

### Writing a New Unit Test

1. Place test in appropriate `packages/*/tests/unit/` directory
2. Use `@pytest.mark.unit` marker
3. Use mocked fixtures: `mock_config`, `mock_model`, `mock_data_loader`
4. Verify test runs in <1 second
5. Ensure test passes with `pytest -m unit`

### Writing a New Integration Test

1. Place test in `tests/integration/`
2. Use `@pytest.mark.integration` marker
3. Add `@pytest.mark.slow` if test takes >10 seconds
4. Add `@pytest.mark.requires_data` if downloading datasets
5. Import real implementations from packages
6. Verify test passes with `pytest -m integration`

## CI/CD Integration

### Fast CI Pipeline (Every Commit)

```yaml
# Run only unit tests
pytest -m unit --cov
```

### Slow CI Pipeline (PR Merge, Nightly)

```yaml
# Run integration tests
pytest -m integration --maxfail=1
```

### Full Test Suite (Release)

```yaml
# Run everything
pytest -m "unit or integration"
```

## Test Coverage

### Current Coverage Goals

- **Unit Tests**: >80% line coverage per package
- **Integration Tests**: Cover all major workflows
- **End-to-End**: At least one test per user-facing feature

### Checking Coverage

```powershell
# Generate coverage report
pytest --cov=packages --cov-report=html

# View report
start htmlcov/index.html
```

## Troubleshooting

### Tests Fail with Import Errors

**Problem**: `ModuleNotFoundError: No module named 'img_classifier_X'`

**Solution**: Ensure package is installed in development mode:
```powershell
uv pip install -e packages/X
```

### Tests Hang or Run Slowly

**Problem**: Tests take too long or hang

**Solutions**:
1. Disable parallel execution: `pytest -n 0`
2. Check for unit tests doing I/O or real training
3. Use mocks instead of real implementations
4. Move slow tests to integration with `@pytest.mark.slow`

### Parallel Test Failures

**Problem**: Tests pass serially but fail in parallel

**Solutions**:
1. Use `isolated_tmp_dir` fixture instead of shared directories
2. Mark test as `@pytest.mark.serial`
3. Avoid global state or shared resources

### Integration Tests Fail

**Problem**: Integration tests fail but unit tests pass

**Solutions**:
1. Check that all packages are properly installed
2. Verify TensorFlow installation: `python -c "import tensorflow"`
3. Ensure test data exists: check `temp_dataset_structure` fixture
4. Run with verbose output: `pytest -m integration -vv`

## Best Practices Summary

### Unit Tests Should Be

- ✅ Fast (<1s per test)
- ✅ Isolated (no external dependencies)
- ✅ Deterministic (same result every time)
- ✅ Independent (can run in any order)
- ✅ Focused (test one thing)

### Integration Tests Should

- ✅ Test real workflows
- ✅ Use real implementations
- ✅ Cover cross-package interactions
- ✅ Be marked appropriately (`@pytest.mark.integration`)
- ✅ Have clear failure messages

### All Tests Should

- ✅ Have descriptive names
- ✅ Have clear docstrings
- ✅ Clean up resources
- ✅ Be maintainable
- ✅ Add value (not just coverage)

## Examples

### Unit Test Example

```python
"""Unit tests for Trainer class."""

import pytest
from unittest.mock import Mock
from img_classifier_training import Trainer


@pytest.mark.unit
class TestTrainer:
    """Unit tests for Trainer with mocked dependencies."""
    
    def test_initialization(self, mock_config, mock_model, mock_data_loader):
        """Test Trainer initializes correctly."""
        trainer = Trainer(mock_config, mock_model, mock_data_loader)
        
        assert trainer.config == mock_config
        assert trainer.model == mock_model
        assert trainer.data_loader == mock_data_loader
    
    def test_train_calls_model_fit(self, mock_config, mock_model, mock_data_loader):
        """Test that train() calls model.fit()."""
        trainer = Trainer(mock_config, mock_model, mock_data_loader)
        
        # Mock data
        X_train, y_train = Mock(), Mock()
        X_val, y_val = Mock(), Mock()
        
        trainer.train(X_train, y_train, X_val, y_val)
        
        # Verify model.fit was called
        mock_model.model.fit.assert_called_once()
```

### Integration Test Example

```python
"""Integration tests for training workflow."""

import pytest
import numpy as np
from img_classifier_config import DatasetConfig
from img_classifier_models import ArchitectureFactory, BaseModel
from img_classifier_training import Trainer


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingWorkflow:
    """Integration tests for complete training workflow."""
    
    def test_end_to_end_training(self, isolated_tmp_dir):
        """Test complete training workflow with real components."""
        # Create real config
        config = DatasetConfig(
            working_dir=isolated_tmp_dir,
            num_classes=2,
            class_names=["class1", "class2"],
            num_epochs=2,  # Minimal for testing
        )
        
        # Create real model
        keras_model = ArchitectureFactory.create(config, complexity="simple")
        model = TestModel(config, keras_model)
        
        # Create synthetic data loader
        loader = create_synthetic_loader(config)
        
        # Create trainer
        trainer = Trainer(config, model, loader)
        
        # Run training
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data()
        model.compile()
        history = trainer.train(X_train, y_train, X_val, y_val)
        
        # Verify results
        assert history is not None
        assert "loss" in history.history
        assert len(history.history["loss"]) == 2
        
        # Evaluate
        loss, acc = trainer.evaluate(X_test, y_test)
        assert 0.0 <= acc <= 1.0
```

## Further Reading

- [pytest documentation](https://docs.pytest.org/)
- [pytest-xdist (parallel execution)](https://pytest-xdist.readthedocs.io/)
- [pytest-cov (coverage)](https://pytest-cov.readthedocs.io/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)

