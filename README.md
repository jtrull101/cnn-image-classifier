# Alzheimer's MRI Neural Network

<div align="center">
  <img src="images/006-11.jpg" alt="Brain" width="80" height="80">
  <br>
  <sub><a href="https://www.vecteezy.com/free-vector/brain">Brain Vectors by Vecteezy</a></sub>
  
  <h3>AI-Powered Alzheimer's Disease Classification</h3>
  <p>Production-ready CNN for analyzing MRI images using a UV-managed Python monorepo</p>
  
  ![Tests](https://github.com/jtrull101/alz-mri-neural-network/actions/workflows/tests.yml/badge.svg)
  [![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
</div>

---

## About

A deep learning system for classifying Alzheimer's disease progression from MRI images, built with modern Python tooling and monorepo best practices. This project was born from a personal mission to combat a disease that has affected my family for generations.

**Classification Categories:**
- No Impairment
- Very Mild Impairment  
- Mild Impairment
- Moderate Impairment

**Important:** Trained on fictitious data from [Kaggle](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy) for educational purposes. Not for medical diagnosis.

---

## Quick Start

### Prerequisites
- Python 3.13+
- [UV](https://github.com/astral-sh/uv) package manager

### Installation

```powershell
# Clone repository
git clone https://github.com/jtrull101/alz-mri-neural-network.git
cd alz-mri-neural-network

# Automated setup
.\scripts\setup.ps1
```

The setup script ensures UV is available, installs dependencies, and runs initial checks/builds.

### Manual Setup

```powershell
# Install UV (if needed) and sync dependencies
make install

# Download datasets (optional - only if you plan to train models)
make download-datasets
```

**Note:** All configuration is now in `pyproject.toml` files. No separate `requirements.txt`, `pytest.ini`, or `setup.cfg` files needed.

### Dataset Download (Optional)

Datasets are hosted on GitHub Releases to keep the repository lightweight (~461 MB total).

```powershell
# Download all datasets
make download-datasets

# List available datasets
make list-datasets

# Download specific dataset
uv run python scripts/download_datasets.py alzheimers-mri-dataset
```

**Available Datasets:**
- `alzheimers-mri-dataset.zip` (72 MB) - Main Alzheimer's classification dataset
- `brain-tumor-mri-dataset.zip` (149 MB) - Brain tumor classification
- `covid-chest-xray.zip` (241 MB) - COVID-19 chest X-ray images

See [docs/DATASETS.md](docs/DATASETS.md) for detailed information.

**Note:** You only need to download datasets if you plan to train models. The pre-trained models and API work without downloading datasets.

---

## NEW: Generalized Image Classification System

This project now includes a **powerful generalized system** that works with ANY image classification dataset, not just Alzheimer's MRI scans!

### Key Features

- **Dataset Auto-Detection** - Automatically analyzes any dataset structure  
- **Dynamic Architecture Generation** - Creates optimal CNN architectures  
- **Hyperparameter Optimization** - Grid, random, or Bayesian search  
- **CLI Interface** - Easy-to-use command-line tools  
- **Python API** - Flexible programmatic access  
- **Full Backward Compatibility** - Existing code still works  

### Quick Start with Any Dataset

```bash
# Install CLI tools
uv pip install -e packages/cli

# Analyze your dataset
img-classifier info /path/to/your/dataset

# Train a model (auto-detects everything!)
img-classifier train /path/to/your/dataset --project-name my_model

# Optimize hyperparameters
img-classifier optimize /path/to/your/dataset --max-trials 20

# Make predictions
img-classifier predict model.keras image.jpg --class-names "cat" "dog" "bird"
```

### Python API

```python
from pathlib import Path
from img_classifier_training import TrainingOrchestrator

# One-liner: auto-detect dataset and train
orchestrator = TrainingOrchestrator.from_dataset_path(
    dataset_path=Path("/path/to/your/dataset"),
    optimize_hyperparameters=True,
    max_trials=20,
)
model = orchestrator.run(plot=True)
```

### Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - 5-minute tutorial
- **[Complete Documentation](docs/GENERALIZED_SYSTEM.md)** - Full system guide
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - From Alzheimer's to generalized
- **[Transformation Summary](docs/TRANSFORMATION_SUMMARY.md)** - What's new
- **[Examples](examples_generalized_system.py)** - Code examples

### Dataset Requirements

Works with any dataset following this structure:

```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── class2/
│   └── class3/
└── test/
    ├── class1/
    ├── class2/
    └── class3/
```

Supports: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`

---

## Original Alzheimer's System

The original Alzheimer's-specific system continues to work as before:

### Usage (Alzheimer's Dataset)

**Note:** All configuration is now in `pyproject.toml` files. No separate `requirements.txt`, `pytest.ini`, or `setup.cfg` files needed.

---

## Monorepo Architecture

This project uses a UV workspace to manage multiple Python packages with shared tooling.

### Structure

```
alz-mri-neural-network/
|-- apps/
|   `-- api/                    # FastAPI application
|-- packages/
|   |-- config/                 # Configuration (Pydantic)
|   |-- utils/                  # Common utilities
|   |-- data/                   # Data loading
|   |-- models/                 # Neural networks
|   |-- training/               # Training pipeline
|   `-- cli/                    # CLI entrypoints
|-- scripts/                    # Dev/CI helper scripts
|-- docs/                       # Documentation
|-- tests/                      # Shared tests
`-- pyproject.toml              # Workspace configuration
```

### Dependency Flow
- `config` -> `utils` -> `data` -> `training` -> `api`
- `models` is consumed by `training` and `api`
- `cli` pulls from config/data/models/training for end-user commands

---

## Usage

### Start the API

```powershell
## Usage

### Start the API (New FastAPI)

```bash
# Start the modern FastAPI server
cd apps/api
python run_api.py

# Or with uvicorn directly
uvicorn img_classifier_api.app:app --reload
```

Navigate to:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### API Endpoints

**GET /** - Modern web interface with drag-and-drop upload

**POST /api/predict** - Classify an image
```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict",
    files={"file": open("image.jpg", "rb")}
)
result = response.json()
print(f"Class: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

**GET /api/models** - List available models
```python
import requests

response = requests.get("http://localhost:8000/api/models")
models = response.json()
print(f"Loaded models: {models['models']}")
```

### Features
- **Multi-Model Support**: Load and switch between models
- **Auto-Discovery**: Finds best available model automatically
- **Modern UI**: Responsive interface with drag-and-drop
- **API Docs**: Auto-generated Swagger/ReDoc documentation
- **Generic**: Works with any image classification model

### Training

```python
from img_classifier_training import Trainer
from img_classifier_models import CnnClassifier
from img_classifier_data import ImageDataLoader
from img_classifier_config import AlzheimerConfig

config = AlzheimerConfig()
model = CnnClassifier(config)
loader = ImageDataLoader(config)
trainer = Trainer(config, model, loader)

accuracy, loss = trainer.run(plot=True)
```

---

## Development

### Commands

```powershell
make help           # Show all commands
make install        # Install UV (if needed) and sync deps
make sync           # Sync deps (including dev)
make build          # Build all packages/apps
make test           # Run all tests with coverage
make test-coverage  # Full coverage reports
make lint           # Ruff lint
make format         # Ruff format
make typecheck      # ty
make spell          # codespell
make audit          # pip-audit (dependency CVEs)
make deadcode       # vulture (dead-code scan)
make security       # audit + spell + deadcode
make serve-api      # Start API
make pre-commit     # Run all checks
```

### Scoped tasks

```powershell
# Build specific package
uv build --directory packages/config
uv build --directory packages/models

# Run targeted tests
uv run python -m pytest packages/data/img_classifier_data/tests -v
.\scripts\run_tests.ps1 -UnitOnly

# Lint/format a path
uv run ruff check apps/api
uv run ruff format packages/training
```

### Adding Dependencies

```powershell
# Add to specific package
cd packages/data
uv add numpy

# Add dev dependency (workspace)
uv add --dev pytest
```

---

## Project Structure

```
alz-mri-neural-network/
|-- apps/
|   `-- api/                    # FastAPI application
|       |-- img_classifier_api/
|       |-- tests/
|       `-- pyproject.toml
|-- packages/
|   |-- config/                 # Configuration
|   |-- utils/                  # Utilities
|   |-- data/                   # Data loading
|   |-- models/                 # Neural networks
|   |-- training/               # Training pipeline
|   `-- cli/                    # CLI interface
|-- docs/                       # System docs
|-- scripts/                    # Developer tooling
|-- tests/                      # Shared tests
|-- pyproject.toml              # UV workspace
|-- Makefile                    # Build commands
`-- README.md                   # This file
```

---

## Technology Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Task Runner** | Make + UV | Workspace orchestration |
| **Package Manager** | UV | 10-100x faster than pip |
| **Linter/Formatter** | Ruff | Replaces flake8, isort, autopep8, bandit, pydocstyle |
| **Type Checker** | ty | Fast, accurate type inference |
| **Testing** | Pytest | With coverage reporting |
| **Security** | pip-audit + gitleaks | Dependency CVE + secrets scanning |
| **Spell/Dead-code** | codespell + vulture | Typo and unused-code detection |
| **Deep Learning** | TensorFlow/Keras | Neural network training |
| **Web Framework** | FastAPI | REST API |
| **Data Processing** | NumPy, OpenCV, Pandas | Image and data manipulation |

---

## Package Details

### config (alz-mri-config)
Configuration management using **Pydantic** for validation and settings.

```python
from img_classifier_config import AlzheimerConfig

# Load from defaults
config = AlzheimerConfig(batch_size=32, num_epochs=25)

# Load from environment variables (IMG_CLASSIFIER_ prefix)
# export IMG_CLASSIFIER_BATCH_SIZE=64
# export IMG_CLASSIFIER_NUM_EPOCHS=50
config = AlzheimerConfig()  # Automatically loads from env

# Load from .env file
config = AlzheimerConfig(_env_file=".env")

# Validation is automatic
config = AlzheimerConfig(batch_size=0)  # Raises ValidationError
```

**Features:**
- Type validation with Pydantic
- Environment variable support
- `.env` file support
- Validation constraints (e.g., batch_size >= 1)
- No external dependencies beyond Pydantic

### utils (alz-mri-utils)
File operations, downloads, archive extraction.

```python
from img_classifier_utils import download_from_google_drive, extract_archive
```

### data (alz-mri-data)
Data loading, preprocessing, caching.

```python
from img_classifier_data import ImageDataLoader
loader = ImageDataLoader(config)
X_train, y_train = loader.load_train_data()
```

### models (alz-mri-models)
Dynamic CNN architecture generation.

```python
from img_classifier_models import ArchitectureFactory

# Auto-generate based on dataset
model = ArchitectureFactory.create(config)

# Or specify complexity
model = ArchitectureFactory.create(config, complexity="deep")

# Use in training
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

### training (alz-mri-training)
Complete training pipeline with optimization.

```python
from img_classifier_training import TrainingOrchestrator

# Simple training
orchestrator = TrainingOrchestrator.from_dataset_path(
    dataset_path=Path("/path/to/dataset")
)
model = orchestrator.run()

# With hyperparameter optimization
orchestrator = TrainingOrchestrator.from_dataset_path(
    dataset_path=Path("/path/to/dataset"),
    optimize_hyperparameters=True,
    max_trials=20
)
model = orchestrator.run()
```

### cli (alz-mri-cli)
Command-line interface for easy access.

```bash
img-classifier train /path/to/dataset
img-classifier optimize /path/to/dataset --max-trials 30
img-classifier predict model.keras image.jpg
```

### api (alz-mri-api)
FastAPI application with web interface.

---

## Testing

```powershell
# Run all tests
make test

# Test specific package
uv run python -m pytest packages/config/img_classifier_config/tests -v
uv run python -m pytest packages/data/img_classifier_data/tests -v

# With coverage
uv run python -m pytest -c pyproject.toml --rootdir . --cov=packages --cov=apps --cov-report=term-missing
```

**Test Suite:** Comprehensive tests covering configuration, data loading, models, training, and integration.

---

## Code Quality

### Standards
- PEP 8 compliance (Ruff)
- Type hints (ty)
- Line length: 100 characters
- Automated formatting

### Run Checks

```powershell
make format        # Auto-format
make lint          # Check issues
make pre-commit    # All checks
```

---

## Troubleshooting

### Module Not Found
```powershell
uv sync
make build
```

### UV issues
```powershell
uv sync --dev --reinstall
```

### Import Errors
Ensure you're using absolute imports:
```python
# ✓ Correct
from img_classifier_config import AlzheimerConfig

# ✗ Wrong
from .config import AlzheimerConfig
```

### TensorFlow Issues
```powershell
uv pip install tensorflow-cpu  # CPU only
```

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Make changes
4. Run checks (`make pre-commit`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing`)
7. Open Pull Request

**Guidelines:**
- Follow existing code style
- Add type hints
- Write tests
- Update documentation

---

## Contact

**Jonathan Trull** - jttrull0@gmail.com

**Project Link:** [https://github.com/jtrull101/alz-mri-neural-network](https://github.com/jtrull101/alz-mri-neural-network)

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555)](https://linkedin.com/in/jonathan--trull)

---

## Acknowledgments

- Dataset: [Best Alzheimer MRI Dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy) (Kaggle)
- Icons: [Vecteezy Brain Vectors](https://www.vecteezy.com/free-vector/brain)
- Tools: [Astral](https://astral.sh/) for UV and Ruff
- Inspiration: My family and all affected by Alzheimer's

---

<div align="center">
  <strong>Made with dedication for Alzheimer's research</strong>
  <br><br>
  Star this repo if you find it helpful!
</div>
