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
make serve-api
# or
npx nx run api:serve
```

Navigate to `http://127.0.0.1:5000`

### API Endpoints

**GET /** - Web interface

**POST /predict** - Predict from MRI image
```python
import requests

response = requests.post(
    "http://127.0.0.1:5000/predict",
    files={"file": open("mri.jpg", "rb")}
)
print(response.json())  # {"class": "No Impairment", "confidence": 0.95}
```

### Training

```python
from alz_mri_training import Trainer
from alz_mri_models import CNNClassifier
from alz_mri_data import ImageDataLoader
from alz_mri_config import AlzheimerConfig

config = AlzheimerConfig()
model = CNNClassifier(config)
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
│       ├── alz_mri_api/
│       │   ├── app.py
│       │   ├── static/         # Model weights
│       │   └── templates/
│       ├── pyproject.toml
│       └── project.json
├── packages/
│   ├── config/                 # Configuration
│   │   ├── alz_mri_config/
│   │   │   ├── base_config.py
│   │   │   └── alzheimer_config.py
│   │   ├── pyproject.toml
│   │   └── project.json
│   ├── utils/                  # Utilities
│   │   └── alz_mri_utils/
│   ├── data/                   # Data loading
│   │   └── alz_mri_data/
│   ├── models/                 # Neural networks
│   │   └── alz_mri_models/
│   └── training/               # Training pipeline
│       └── alz_mri_training/
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
from alz_mri_config import AlzheimerConfig

# Load from defaults
config = AlzheimerConfig(batch_size=32, num_epochs=25)

# Load from environment variables (ALZ_MRI_ prefix)
# export ALZ_MRI_BATCH_SIZE=64
# export ALZ_MRI_NUM_EPOCHS=50
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
from alz_mri_utils import download_from_google_drive, extract_archive
```

### data (alz-mri-data)
Data loading, preprocessing, caching.

```python
from alz_mri_data import ImageDataLoader
loader = ImageDataLoader(config)
X_train, y_train = loader.load_train_data()
```

### models (alz-mri-models)
CNN architectures (CNNClassifier, SimpleCNN).

```python
from alz_mri_models import CNNClassifier
model = CNNClassifier(config)
model.compile()
```

### training (alz-mri-training)
Training pipeline with callbacks and visualization.

```python
from alz_mri_training import Trainer
trainer = Trainer(config, model, loader)
trainer.run()
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

**Test Suite:** 73 tests covering configuration, data loading, models, training, and integration.

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
from alz_mri_config import AlzheimerConfig

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
