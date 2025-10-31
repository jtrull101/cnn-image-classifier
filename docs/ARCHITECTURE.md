# Architecture Documentation

## Overview
This document describes the rearchitected codebase for the Alzheimer's MRI Neural Network project.

## Architectural Principles

The rearchitected codebase follows these key principles:

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Abstraction**: Base classes define interfaces that can be extended
3. **Configurability**: Configuration is centralized and type-safe using dataclasses
4. **Testability**: Components are designed to be easily tested in isolation
5. **Reusability**: Common functionality is extracted into utility modules

## Module Structure

```
src/alz_mri_cnn/
├── config/              # Configuration management
│   ├── __init__.py
│   ├── base_config.py   # Base configuration class
│   └── alzheimer_config.py  # Alzheimer-specific configuration
├── data/                # Data loading and preprocessing
│   ├── __init__.py
│   ├── base_loader.py   # Abstract data loader interface
│   └── image_loader.py  # Image data loader implementation
├── models/              # Model architectures
│   ├── __init__.py
│   ├── base_model.py    # Abstract model interface
│   └── cnn_classifier.py  # CNN model implementations
├── training/            # Training pipeline
│   ├── __init__.py
│   ├── callbacks.py     # Custom training callbacks
│   └── trainer.py       # Training orchestration
└── utils/               # Utility functions
    ├── __init__.py
    └── file_utils.py    # File operations utilities
```

## Component Details

### Configuration Module (`config/`)

**Purpose**: Centralize all configuration parameters for the project.

**Components**:
- `BaseConfig`: Base configuration class with common parameters
- `AlzheimerConfig`: Alzheimer-specific configuration extending BaseConfig

**Key Features**:
- Type-safe configuration using Python dataclasses
- Automatic path initialization
- Derived properties for common paths (models_dir, logs_dir, cache_dir)
- Easy to extend for new datasets or model architectures

**Usage Example**:
```python
from src.alz_mri_cnn.config import AlzheimerConfig

config = AlzheimerConfig(
    batch_size=32,
    num_epochs=25,
    learning_rate=0.001
)
config.create_directories()
```

### Data Module (`data/`)

**Purpose**: Handle all data loading, preprocessing, and caching.

**Components**:
- `BaseDataLoader`: Abstract base class defining the data loader interface
- `ImageDataLoader`: Concrete implementation for image datasets

**Key Features**:
- Abstract interface allows easy swapping of data sources
- Automatic dataset download from Google Drive
- Data caching for faster subsequent loads
- Configurable data splitting and reduction
- Support for multiple image formats

**Usage Example**:
```python
from src.alz_mri_cnn.data import ImageDataLoader
from src.alz_mri_cnn.config import AlzheimerConfig

config = AlzheimerConfig()
loader = ImageDataLoader(config)

# Setup and load data
loader.setup()
X_train, y_train = loader.load_train_data()
X_test, y_test = loader.load_test_data()
```

### Models Module (`models/`)

**Purpose**: Define and manage model architectures.

**Components**:
- `BaseModel`: Abstract base class for all models
- `CNNClassifier`: Convolutional neural network implementation
- `SimpleCNN`: Lighter CNN for quick experiments

**Key Features**:
- Consistent interface for all models
- Easy model compilation and training
- Model persistence (save/load)
- Reproducible results via seed management

**Usage Example**:
```python
from src.alz_mri_cnn.models import CNNClassifier
from src.alz_mri_cnn.config import AlzheimerConfig

config = AlzheimerConfig()
model = CNNClassifier(config)
model.compile()
model.summary()
```

### Training Module (`training/`)

**Purpose**: Orchestrate the training process.

**Components**:
- `Trainer`: Main training pipeline orchestrator
- `AccuracyThresholdCallback`: Custom callback to stop training at target accuracy

**Key Features**:
- Complete training pipeline management
- Automatic data preparation and splitting
- Configurable callbacks (early stopping, checkpointing)
- Training history visualization
- Automatic results logging

**Usage Example**:
```python
from src.alz_mri_cnn.training import Trainer
from src.alz_mri_cnn.models import CNNClassifier
from src.alz_mri_cnn.data import ImageDataLoader
from src.alz_mri_cnn.config import AlzheimerConfig

config = AlzheimerConfig()
model = CNNClassifier(config)
data_loader = ImageDataLoader(config)

trainer = Trainer(config, model, data_loader)
acc, loss = trainer.run(plot=True, force_save=False)
```

### Utils Module (`utils/`)

**Purpose**: Provide common utility functions.

**Components**:
- `file_utils.py`: File and directory operations

**Key Features**:
- Google Drive download functionality
- Archive extraction (zip and other formats)
- Directory management
- Dataset organization

## Design Patterns Used

### 1. Abstract Base Class (ABC) Pattern
- `BaseModel`, `BaseDataLoader`, `BaseConfig`
- Ensures consistent interfaces across implementations
- Makes it easy to add new model types or data sources

### 2. Strategy Pattern
- Different data loaders can be swapped without changing training code
- Different models can be used with the same training pipeline

### 3. Template Method Pattern
- `Trainer` defines the training workflow
- Specific steps can be customized via configuration

### 4. Factory Pattern (Implicit)
- Configuration classes act as factories for creating properly configured objects
- Ensures all components use consistent settings

## Benefits of the New Architecture

### 1. Maintainability
- Clear module boundaries
- Single responsibility principle
- Easy to locate and fix bugs

### 2. Extensibility
- Add new models by extending `BaseModel`
- Add new data sources by extending `BaseDataLoader`
- Add new configurations by extending `BaseConfig`

### 3. Testability
- Each component can be tested in isolation
- Mock objects can easily replace real implementations
- Test coverage is comprehensive (73 tests)

### 4. Reusability
- Components can be reused across different projects
- Configuration system is generic and adaptable
- Data loading pipeline works for any image classification task

### 5. Type Safety
- Python type hints throughout
- Dataclasses provide runtime type checking
- Reduces runtime errors

## Migration Guide

### From Old Code to New Architecture

**Old Way** (monolithic):
```python
# Everything in one place
init()
load_data()
train_model(epochs=25)
```

**New Way** (modular):
```python
from src.alz_mri_cnn.config import AlzheimerConfig
from src.alz_mri_cnn.data import ImageDataLoader
from src.alz_mri_cnn.models import CNNClassifier
from src.alz_mri_cnn.training import Trainer

# Configure
config = AlzheimerConfig(num_epochs=25)

# Load data
data_loader = ImageDataLoader(config)
data_loader.setup()

# Create model
model = CNNClassifier(config)

# Train
trainer = Trainer(config, model, data_loader)
acc, loss = trainer.run()
```

## Future Enhancements

1. **Add More Data Loaders**: Support for medical imaging formats (DICOM, NIfTI)
2. **Add More Models**: Transfer learning models (ResNet, VGG, EfficientNet)
3. **Add Data Augmentation**: Built-in augmentation pipeline
4. **Add Experiment Tracking**: Integration with MLflow or Weights & Biases
5. **Add Model Interpretability**: Grad-CAM and attention visualizations
6. **Add Distributed Training**: Multi-GPU and multi-node support
