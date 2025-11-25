# Alzheimer's MRI Neural Network

<div align="center">
  <img src="images/006-11.jpg" alt="Brain" width="80" height="80">
  <br>
  <sub><a href="https://www.vecteezy.com/free-vector/brain">Brain Vectors by Vecteezy</a></sub>
  
  <h3>AI-Powered Alzheimer's Disease Classification</h3>
  <p>Production-ready CNN for analyzing MRI images using NX/UV monorepo architecture</p>
  
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

**⚠️ Important:** Trained on fictitious data from [Kaggle](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy) for educational purposes. Not for medical diagnosis.

---

## Quick Start

### Prerequisites
- Python 3.13+
- Node.js 18+ (for NX)
- [UV](https://github.com/astral-sh/uv) package manager

### Installation

```powershell
# Clone repository
git clone https://github.com/jtrull101/alz-mri-neural-network.git
cd alz-mri-neural-network

# Automated setup
.\scripts\setup_monorepo.ps1
```

The setup script installs NX, UV dependencies, and builds all packages.

### Manual Setup

```powershell
# Install UV
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install dependencies
npm install       # NX
uv sync          # Python packages

# Build packages
make build
```

**Note:** All configuration is now in `pyproject.toml` files. No separate `requirements.txt`, `pytest.ini`, or `setup.cfg` files needed.

---

## 🚀 NEW: Generalized Image Classification System

This project now includes a **powerful generalized system** that works with ANY image classification dataset, not just Alzheimer's MRI scans!

### Key Features

✅ **Dataset Auto-Detection** - Automatically analyzes any dataset structure  
✅ **Dynamic Architecture Generation** - Creates optimal CNN architectures  
✅ **Hyperparameter Optimization** - Grid, random, or Bayesian search  
✅ **CLI Interface** - Easy-to-use command-line tools  
✅ **Python API** - Flexible programmatic access  
✅ **Full Backward Compatibility** - Existing code still works  

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

- 📘 **[Quick Start Guide](docs/QUICK_START.md)** - 5-minute tutorial
- 📚 **[Complete Documentation](docs/GENERALIZED_SYSTEM.md)** - Full system guide
- 🔄 **[Migration Guide](docs/MIGRATION_GUIDE.md)** - From Alzheimer's to generalized
- 📊 **[Transformation Summary](docs/TRANSFORMATION_SUMMARY.md)** - What's new
- 💻 **[Examples](examples_generalized_system.py)** - Code examples

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

This project uses an **NX/UV managed monorepo** for better modularity and dependency management.

### Structure

```
apps/
  api/              # Flask REST API
packages/
  config/           # Configuration (no dependencies)
  utils/            # Common utilities
  data/             # Data loading → config, utils
  models/           # Neural networks → config
  training/         # Training pipeline → config, data, models
```

### Dependency Graph

```
config (base) → utils → data → training → api
                    ↓      ↓       ↓
                 models ────────────
```

### Benefits
- **Modularity**: Clear package boundaries
- **Speed**: Parallel builds, incremental rebuilds, caching
- **Type Safety**: Per-package type checking
- **Reusability**: Packages work independently

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
- ✅ **Multi-Model Support**: Load and switch between models
- ✅ **Auto-Discovery**: Finds best available model automatically
- ✅ **Modern UI**: Responsive interface with drag-and-drop
- ✅ **API Docs**: Auto-generated Swagger/ReDoc documentation
- ✅ **Generic**: Works with any image classification model

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
make help          # Show all commands
make build         # Build all packages
make test          # Run all tests
make lint          # Lint code
make format        # Format code
make serve-api     # Start API
make graph         # View dependency graph
make pre-commit    # Run all checks
```

### NX Commands

```powershell
# Build specific package
npx nx run config:build
npx nx run data:build

# Test specific package  
npx nx run models:test

# Lint/format specific package
npx nx run api:lint
npx nx run training:format

# Only affected by changes
npx nx affected --target=build
npx nx affected --target=test
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
├── apps/
│   └── api/                    # Flask application
│       ├── img_classifier_api/
│       │   ├── app.py
│       │   ├── static/         # Model weights
│       │   └── templates/
│       ├── pyproject.toml
│       └── project.json
├── packages/
│   ├── config/                 # Configuration
│   │   ├── img_classifier_config/
│   │   │   ├── base_config.py
│   │   │   └── alzheimer_config.py
│   │   ├── pyproject.toml
│   │   └── project.json
│   ├── utils/                  # Utilities
│   │   └── img_classifier_utils/
│   ├── data/                   # Data loading
│   │   └── img_classifier_data/
│   ├── models/                 # Neural networks
│   │   └── img_classifier_models/
│   └── training/               # Training pipeline
│       └── img_classifier_training/
├── docs/
│   ├── ARCHITECTURE.md         # System design
│   └── TESTING.md              # Test strategy
├── nx.json                     # NX config
├── package.json                # Node/NX deps
├── pyproject.toml              # UV workspace
├── Makefile                    # Build commands
└── README.md                   # This file
```

---

## Technology Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Build System** | NX | Task orchestration, caching, parallel builds |
| **Package Manager** | UV | 10-100x faster than pip |
| **Linter/Formatter** | Ruff | Replaces flake8, isort, autopep8 |
| **Type Checker** | Pyright | Fast, accurate type inference |
| **Testing** | Pytest | With coverage reporting |
| **Deep Learning** | TensorFlow/Keras | Neural network training |
| **Web Framework** | Flask | REST API |
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
Flask REST API with web interface.

---

## Testing

```powershell
# Run all tests
make test

# Test specific package
npx nx run config:test
npx nx run data:test

# With coverage
uv run pytest --cov
```

**Test Suite:** Comprehensive tests covering configuration, data loading, models, training, and integration.

---

## Code Quality

### Standards
- PEP 8 compliance (Ruff)
- Type hints (Pyright)
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

### NX Commands Fail
```powershell
npm install
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

## License

MIT License - see [LICENSE](LICENSE) file.

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
  <strong>Made with ❤️ and 🧠 for Alzheimer's research</strong>
  <br><br>
  ⭐ Star this repo if you find it helpful!
</div>
