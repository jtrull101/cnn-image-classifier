# Quick Start Guide: Generalized Image Classification

Get up and running with the generalized image classification system in minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/jtrull101/alz-mri-neural-network.git
cd alz-mri-neural-network

# Install dependencies
uv sync

# Install CLI tools
cd packages/cli
uv pip install -e .
cd ../..
```

## 5-Minute Tutorial

### Step 1: Prepare Your Dataset

Organize your images in this structure:

```
my_dataset/
├── train/
│   ├── cat/
│   │   ├── cat1.jpg
│   │   ├── cat2.jpg
│   │   └── ...
│   ├── dog/
│   │   └── ...
│   └── bird/
│       └── ...
└── test/
    ├── cat/
    ├── dog/
    └── bird/
```

### Step 2: Analyze Your Dataset

```bash
img-classifier info /path/to/my_dataset
```

Output:
```
============================================================
Dataset Information
============================================================
Path: /path/to/my_dataset
Number of classes: 3
Class names: bird, cat, dog
Total images: 1500

Class distribution:
  bird: 500
  cat: 500
  dog: 500

Balanced: Yes
Train/Test split: Yes
Recommended complexity: medium
Sample image shape: 224x224x3
============================================================
```

### Step 3: Train Your First Model

```bash
img-classifier train /path/to/my_dataset \
  --project-name my_first_model \
  --epochs 20 \
  --plot
```

The system will:
1. ✅ Auto-detect dataset characteristics
2. ✅ Generate appropriate CNN architecture
3. ✅ Train the model
4. ✅ Save the best model
5. ✅ Show training plots

### Step 4: Make Predictions

```bash
img-classifier predict \
  /tmp/my_first_model_classifier/models/best_model.keras \
  my_test_image.jpg \
  --class-names "bird" "cat" "dog"
```

Output:
```
============================================================
Prediction Result
============================================================
Predicted class: cat
Confidence: 94.52%

All probabilities:
  bird: 2.31%
  cat: 94.52%
  dog: 3.17%
============================================================
```

## Common Use Cases

### Use Case 1: Quick Experiment

Test quickly with 10% of your data:

```bash
img-classifier train /path/to/dataset \
  --project-name quick_test \
  --architecture simple \
  --epochs 10 \
  --batch-size 64
```

Then modify the config programmatically:

```python
from pathlib import Path
from img_classifier_config import DatasetDetector

detector = DatasetDetector(Path("/path/to/dataset"))
config = detector.create_config(
    project_name="quick_test",
    data_percent=0.1,  # Use only 10%
    num_epochs=10,
    architecture_complexity="simple"
)
config.to_yaml("quick_config.yaml")
```

### Use Case 2: Find Best Hyperparameters

```bash
img-classifier optimize /path/to/dataset \
  --optimizer random \
  --max-trials 30 \
  --target-accuracy 0.95 \
  --quick
```

Results saved to `/tmp/{dataset}_classifier/logs/optimization/optimization_results.json`

### Use Case 3: Production Training

1. Create production config:

```bash
img-classifier create-config /path/to/dataset production.yaml \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --architecture deep
```

2. Train with config:

```bash
img-classifier train /path/to/dataset --config production.yaml --plot
```

### Use Case 4: Python API

```python
from pathlib import Path
from img_classifier_training import TrainingOrchestrator

# One-liner training
orchestrator = TrainingOrchestrator.from_dataset_path(
    dataset_path=Path("/path/to/dataset"),
    project_name="my_model",
)
model = orchestrator.run(plot=True)
```

## Architecture Selection

The system automatically selects architecture based on your dataset:

| Dataset Size | Classes | Recommended Architecture | Training Time |
|--------------|---------|-------------------------|---------------|
| < 1,000 images | 2-3 | Simple | 5-10 min |
| 1,000-10,000 | 4-10 | Medium | 20-60 min |
| > 10,000 | 10+ | Deep | 1-4 hours |

Override with `--architecture` flag:

```bash
img-classifier train /path/to/dataset --architecture deep
```

## Optimization Strategies

### Random Search (Recommended for most cases)

```bash
img-classifier optimize /path/to/dataset \
  --optimizer random \
  --max-trials 20
```

- ⚡ Fast
- 🎯 Good results
- 🔀 Explores search space well

### Grid Search (Exhaustive)

```bash
img-classifier optimize /path/to/dataset \
  --optimizer grid \
  --quick  # Reduces search space
```

- 🐌 Slow
- ✅ Guaranteed to find best in search space
- 💰 Resource intensive

### Bayesian Optimization (Best results)

```bash
# Requires: uv pip install optuna
img-classifier optimize /path/to/dataset \
  --optimizer bayesian \
  --max-trials 50
```

- 🧠 Learns from previous trials
- 🎯 Most efficient
- ⏱️ Best for expensive evaluations

## Configuration Files

### Generate Template

```bash
img-classifier create-config /path/to/dataset my_config.yaml
```

### Edit Configuration

```yaml
# my_config.yaml
project_name: "my_classifier"
dataset_name: "my_dataset"
data_path: "/path/to/dataset"

# Image settings
image_size: [128, 128]
color_channels: 3

# Training
batch_size: 32
num_epochs: 50
learning_rate: 0.001

# Architecture
architecture_complexity: "medium"
dropout_rate: 0.3
```

### Use Configuration

```bash
img-classifier train /path/to/dataset --config my_config.yaml
```

## Tips for Better Results

### 1. Start Small, Scale Up

```bash
# First: Quick test
img-classifier train /path/to/dataset \
  --architecture simple \
  --epochs 5

# If promising: Full training
img-classifier train /path/to/dataset \
  --architecture medium \
  --epochs 50 \
  --plot

# For production: Optimize
img-classifier optimize /path/to/dataset \
  --max-trials 30
```

### 2. Monitor Training

Use `--plot` flag to visualize training:

```bash
img-classifier train /path/to/dataset --plot
```

Watch for:
- 📈 Val accuracy increasing
- 📉 Val loss decreasing
- ⚠️ Overfitting (val loss increases while train loss decreases)

### 3. Handle Overfitting

If validation accuracy plateaus or decreases:

```python
# Increase dropout
config.dropout_rate = 0.5

# Use data augmentation (future feature)
# config.use_augmentation = True

# Reduce model complexity
config.architecture_complexity = "simple"
```

### 4. Speed Up Training

```python
# Use smaller images
config.image_size = (64, 64)

# Larger batches (if GPU memory allows)
config.batch_size = 64

# Use subset of data for experiments
config.data_percent = 0.1
```

## Common Issues

### Issue: Out of Memory

```bash
# Reduce batch size
img-classifier train /path/to/dataset --batch-size 16

# Or reduce image size in config
```

### Issue: Poor Accuracy

```bash
# Try optimization
img-classifier optimize /path/to/dataset --max-trials 30

# Or use deeper architecture
img-classifier train /path/to/dataset --architecture deep --epochs 100
```

### Issue: Training Too Slow

```bash
# Use simpler architecture
img-classifier train /path/to/dataset --architecture simple

# Use less data
img-classifier train /path/to/dataset  # then edit config: data_percent: 0.1
```

## Next Steps

1. **Read Full Documentation**: `docs/GENERALIZED_SYSTEM.md`
2. **Try Examples**: `python examples_generalized_system.py`
3. **Explore CLI**: `img-classifier --help`
4. **Migration Guide**: `docs/MIGRATION_GUIDE.md` (if coming from old system)

## Getting Help

```bash
# General help
img-classifier --help

# Command-specific help
img-classifier train --help
img-classifier optimize --help
img-classifier info --help
```

Or check:
- 📚 `docs/GENERALIZED_SYSTEM.md` - Full documentation
- 🔄 `docs/MIGRATION_GUIDE.md` - Migration from old system
- 💻 `examples_generalized_system.py` - Code examples
- 🐛 GitHub Issues - Report bugs or ask questions

## Cheat Sheet

```bash
# Analyze dataset
img-classifier info /path/to/dataset

# Quick training
img-classifier train /path/to/dataset

# Custom training
img-classifier train /path/to/dataset \
  --epochs 50 --batch-size 32 --architecture medium

# Optimize hyperparameters
img-classifier optimize /path/to/dataset --max-trials 20

# Create config
img-classifier create-config /path/to/dataset config.yaml

# Train with config
img-classifier train /path/to/dataset --config config.yaml

# Predict
img-classifier predict model.keras image.jpg \
  --class-names "class1" "class2" "class3"
```

Happy training! 🚀

