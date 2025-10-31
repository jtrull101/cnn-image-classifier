# Testing Documentation

## Overview
This document describes the comprehensive test suite created for the rearchitected Alzheimer's MRI Neural Network codebase.

## Test Structure

### Test Files
- `test_config.py` - Tests for configuration management
- `test_utils.py` - Tests for utility functions
- `test_data_loaders.py` - Tests for data loading components
- `test_models.py` - Tests for model architecture
- `test_training.py` - Tests for training pipeline
- `test_integration.py` - Integration tests for overall architecture

## Test Coverage

### Configuration Module (`test_config.py`)
**14 tests - All passing ✓**

Tests cover:
- Default configuration values
- Custom configuration parameters
- Path initialization and management
- Directory creation
- Input shape calculation
- Alzheimer-specific configurations
- Class name management

Key test cases:
- `test_default_values()` - Verifies default settings
- `test_create_directories()` - Ensures directories are created properly
- `test_input_shape()` - Validates input shape property
- `test_class_names()` - Verifies Alzheimer class definitions

### Utilities Module (`test_utils.py`)
**7 tests - All passing ✓**

Tests cover:
- Directory creation with nested paths
- File removal with pattern matching
- Handling of existing directories
- Edge cases (non-existent directories, empty directories)

Key test cases:
- `test_ensure_directory_exists_creates_nested_directories()` - Tests deep path creation
- `test_clean_directory_removes_files()` - Validates file cleanup
- `test_clean_directory_preserves_subdirectories()` - Ensures subdirs aren't deleted

### Data Loaders Module (`test_data_loaders.py`)
**15 tests - All passing ✓**

Tests cover:
- Abstract base class interface
- Category detection and listing
- Data splitting for train/val/test
- Dataset size reduction
- Cache path generation
- Image data loader initialization
- Dataset preparation workflow

Key test cases:
- `test_split_data()` - Validates data splitting logic
- `test_reduce_dataset()` - Tests dataset reduction
- `test_get_categories()` - Verifies category detection
- `test_prepare_dataset_already_prepared()` - Tests idempotency

### Models Module (`test_models.py`)
**13 tests - Skipped (requires TensorFlow)**

Tests cover:
- Model building and compilation
- Layer architecture
- Input/output shapes
- Model persistence
- Random seed management

Key test cases (will run when TensorFlow is available):
- `test_build_creates_model()` - Verifies model creation
- `test_model_accepts_correct_input_shape()` - Validates input handling
- `test_output_is_probability_distribution()` - Checks softmax output

### Training Module (`test_training.py`)
**18 tests - Skipped (requires TensorFlow)**

Tests cover:
- Callback functionality
- Training pipeline
- Data preparation
- Model evaluation
- Results logging
- Model checkpointing

Key test cases (will run when TensorFlow is available):
- `test_train_returns_history()` - Validates training execution
- `test_evaluate()` - Tests model evaluation
- `test_save_model_requires_minimum_accuracy()` - Checks save logic

### Integration Tests (`test_integration.py`)
**6 tests - All passing ✓**

Tests cover:
- Module import verification
- Component integration
- Config and data loader interaction
- Module structure validation

Key test cases:
- `test_config_and_data_loader_integration()` - Tests component interaction
- `test_module_structure()` - Validates package organization

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Files
```bash
python -m pytest tests/test_config.py -v
python -m pytest tests/test_utils.py -v
python -m pytest tests/test_data_loaders.py -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=src/alz_mri_cnn --cov-report=html
```

## Test Results Summary

| Module | Tests | Passing | Skipped | Status |
|--------|-------|---------|---------|--------|
| Config | 14 | 14 | 0 | ✓ All passing |
| Utils | 7 | 7 | 0 | ✓ All passing |
| Data Loaders | 15 | 15 | 0 | ✓ All passing |
| Integration | 6 | 6 | 0 | ✓ All passing |
| Models | 13 | 0 | 13 | ⏸ Requires TensorFlow |
| Training | 18 | 0 | 18 | ⏸ Requires TensorFlow |
| **Total** | **73** | **42** | **31** | **57% passing** |

## Dependencies for Full Test Suite

To run all tests including the skipped ones:

```bash
pip install tensorflow>=2.12 keras
```

Note: The skipped tests require TensorFlow to be installed. Once installed, all 73 tests should pass.

## Test Design Principles

1. **Isolation**: Each test is independent and uses temporary directories
2. **Clarity**: Test names clearly describe what is being tested
3. **Coverage**: Tests cover happy paths, edge cases, and error conditions
4. **Maintainability**: Tests are organized by module and easy to update
5. **Speed**: Tests without heavy dependencies run quickly (<1 second)

## Continuous Integration

These tests are designed to run in CI/CD pipelines. The test suite:
- Completes in under 1 second for core modules
- Uses mock objects where appropriate
- Cleans up all temporary resources
- Provides clear failure messages

## Future Test Additions

Consider adding:
1. Performance benchmarks for data loading
2. End-to-end training tests (requires sample data)
3. Model accuracy validation tests
4. Memory usage tests for large datasets
5. Cross-platform compatibility tests
