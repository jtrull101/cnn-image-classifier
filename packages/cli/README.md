# CLI Package

Command-line interface for the image classification neural network system.

## Installation

```bash
uv pip install -e packages/cli
```

## Usage

### Display dataset information

```bash
img-classifier info path/to/dataset
```

### Train a model

```bash
img-classifier train path/to/dataset --epochs 30 --batch-size 32
```

### Optimize hyperparameters

```bash
img-classifier optimize path/to/dataset --optimizer random --max-trials 20
```

### Create a configuration file

```bash
img-classifier create-config path/to/dataset config.yaml
```

### Make predictions

```bash
img-classifier predict model.keras image.jpg --class-names "Class1" "Class2" "Class3"
```

## Commands

- `info`: Display dataset information
- `train`: Train a model on a dataset
- `optimize`: Run hyperparameter optimization
- `create-config`: Generate a configuration file
- `predict`: Make predictions on images

