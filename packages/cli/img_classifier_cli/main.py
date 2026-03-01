"""Command-line interface for the image classification system."""

import sys
from pathlib import Path
from typing import Optional

import click
from img_classifier_config import DatasetConfig, DatasetDetector
from img_classifier_training import HyperparameterSpace, TrainingOrchestrator


_SEPARATOR = "=" * 60


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Image Classification Neural Network System.

    A modular, automated system for training CNN models on any image
    classification dataset with hyperparameter optimization.
    """
    pass


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option("--project-name", "-n", help="Name for the project")
@click.option("--working-dir", "-w", type=click.Path(path_type=Path), help="Working directory")
def info(dataset_path: Path, project_name: Optional[str], working_dir: Optional[Path]) -> None:
    """Display information about a dataset."""
    detector = DatasetDetector(dataset_path)
    info_dict = detector.detect()

    click.echo("\n" + _SEPARATOR)
    click.echo("Dataset Information")
    click.echo(_SEPARATOR)
    click.echo(f"Path: {info_dict['dataset_path']}")
    click.echo(f"Number of classes: {info_dict['num_classes']}")
    click.echo(f"Class names: {', '.join(info_dict['class_names'])}")
    click.echo(f"Total images: {info_dict['total_images']}")
    click.echo("\nClass distribution:")
    for cls, count in info_dict["class_distribution"].items():
        click.echo(f"  {cls}: {count}")
    click.echo(f"\nBalanced: {'Yes' if info_dict['is_balanced'] else 'No'}")
    click.echo(f"Train/Test split: {'Yes' if info_dict['has_train_test_split'] else 'No'}")
    click.echo(f"Recommended complexity: {info_dict['recommended_complexity']}")
    if info_dict["sample_image_shape"]:
        h, w, c = info_dict["sample_image_shape"]
        click.echo(f"Sample image shape: {h}x{w}x{c}")
    click.echo(_SEPARATOR + "\n")


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option("--project-name", "-n", help="Name for the project")
@click.option("--working-dir", "-w", type=click.Path(path_type=Path), help="Working directory")
@click.option("--config", "-c", type=click.Path(path_type=Path), help="Path to config YAML file")
@click.option("--epochs", "-e", type=int, help="Number of training epochs")
@click.option("--batch-size", "-b", type=int, help="Batch size")
@click.option("--learning-rate", "-lr", type=float, help="Learning rate")
@click.option(
    "--architecture",
    "-a",
    type=click.Choice(["auto", "simple", "medium", "deep"]),
    help="Model architecture complexity",
)
@click.option("--plot/--no-plot", default=False, help="Plot training history")
def train(
    dataset_path: Path,
    project_name: Optional[str],
    working_dir: Optional[Path],
    config: Optional[Path],
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    architecture: Optional[str],
    plot: bool,
) -> None:
    """Train a model on a dataset."""
    click.echo("\n" + _SEPARATOR)
    click.echo("Training Neural Network")
    click.echo(_SEPARATOR + "\n")

    # Load or create config
    if config:
        cfg = DatasetConfig.from_yaml(config)
    else:
        detector = DatasetDetector(dataset_path)
        cfg = detector.create_config(project_name=project_name, working_dir=working_dir)

    # Override with command-line arguments
    if epochs:
        cfg.num_epochs = epochs
    if batch_size:
        cfg.batch_size = batch_size
    if learning_rate:
        cfg.learning_rate = learning_rate
    if architecture:
        cfg.architecture_complexity = architecture

    # Create orchestrator and run
    orchestrator = TrainingOrchestrator(cfg, optimize_hyperparameters=False)

    try:
        orchestrator.run(plot=plot)
        click.echo("\n" + _SEPARATOR)
        click.echo("Training completed successfully!")
        click.echo(_SEPARATOR + "\n")
    except KeyboardInterrupt:
        click.echo("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\nError during training: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option("--project-name", "-n", help="Name for the project")
@click.option("--working-dir", "-w", type=click.Path(path_type=Path), help="Working directory")
@click.option("--config", "-c", type=click.Path(path_type=Path), help="Path to config YAML file")
@click.option(
    "--optimizer",
    "-o",
    "optimizer_type",
    type=click.Choice(["random", "grid", "bayesian"]),
    default="random",
    help="Optimization algorithm",
)
@click.option("--max-trials", "-t", type=int, default=20, help="Maximum number of trials")
@click.option("--target-accuracy", type=float, default=0.95, help="Target accuracy to stop early")
@click.option("--quick/--full", default=False, help="Use quick search space")
def optimize(
    dataset_path: Path,
    project_name: Optional[str],
    working_dir: Optional[Path],
    config: Optional[Path],
    optimizer_type: str,
    max_trials: int,
    target_accuracy: float,
    quick: bool,
) -> None:
    """Optimize hyperparameters for a dataset."""
    click.echo("\n" + _SEPARATOR)
    click.echo("Hyperparameter Optimization")
    click.echo(_SEPARATOR)
    click.echo(f"Optimizer: {optimizer_type}")
    click.echo(f"Max trials: {max_trials}")
    click.echo(f"Target accuracy: {target_accuracy}")
    click.echo(_SEPARATOR + "\n")

    # Load or create config
    if config:
        cfg = DatasetConfig.from_yaml(config)
    else:
        detector = DatasetDetector(dataset_path)
        cfg = detector.create_config(project_name=project_name, working_dir=working_dir)

    # Create search space
    search_space = HyperparameterSpace.quick() if quick else HyperparameterSpace.default()

    # Create orchestrator and run
    orchestrator: TrainingOrchestrator = TrainingOrchestrator(
        cfg,
        optimize_hyperparameters=True,
        optimizer_type=optimizer_type,
        search_space=search_space,
        max_trials=max_trials,
    )

    try:
        orchestrator.run(plot=False)
        click.echo("\n" + _SEPARATOR)
        click.echo("Optimization completed successfully!")
        click.echo(_SEPARATOR)

        opt = orchestrator.optimizer
        if opt is not None:
            click.echo(opt.get_summary())

    except KeyboardInterrupt:
        click.echo("\nOptimization interrupted by user.")
        opt = orchestrator.optimizer
        if opt is not None and opt.results:
            click.echo("\nPartial results:")
            click.echo(opt.get_summary())
        sys.exit(1)
    except Exception as e:
        click.echo(f"\nError during optimization: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--project-name", "-n", help="Name for the project")
@click.option("--epochs", "-e", type=int, default=25, help="Number of training epochs")
@click.option("--batch-size", "-b", type=int, default=32, help="Batch size")
@click.option("--learning-rate", "-lr", type=float, default=0.001, help="Learning rate")
@click.option(
    "--architecture",
    "-a",
    type=click.Choice(["auto", "simple", "medium", "deep"]),
    default="auto",
    help="Model architecture complexity",
)
def create_config(
    dataset_path: Path,
    output_path: Path,
    project_name: Optional[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    architecture: str,
) -> None:
    """Create a configuration file for a dataset."""
    detector = DatasetDetector(dataset_path)
    cfg = detector.create_config(
        project_name=project_name,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        architecture_complexity=architecture,
    )

    cfg.to_yaml(output_path)
    click.echo(f"\nConfiguration saved to: {output_path}")
    click.echo("\nYou can now use this config with:")
    click.echo(f"  img-classifier train {dataset_path} --config {output_path}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--class-names", "-c", multiple=True, help="Class names (in order)")
def predict(model_path: Path, image_path: Path, class_names: tuple) -> None:
    """Make a prediction on a single image."""
    import cv2
    import numpy as np
    import tensorflow as tf

    # Load model
    click.echo(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Get input shape
    input_shape = model.input_shape[1:3]  # type: ignore

    # Load and preprocess image
    click.echo(f"Loading image from {image_path}...")
    image = cv2.imread(str(image_path))
    if image is None:
        click.echo(f"Error: Could not load image from {image_path}", err=True)
        sys.exit(1)

    image_resized = cv2.resize(image, input_shape)
    image_array = np.asarray(image_resized, dtype=float) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Make prediction
    click.echo("Making prediction...")
    predictions = model.predict(image_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Display result
    click.echo("\n" + _SEPARATOR)
    click.echo("Prediction Result")
    click.echo(_SEPARATOR)
    if class_names and len(class_names) > predicted_class:
        click.echo(f"Predicted class: {class_names[predicted_class]}")
    else:
        click.echo(f"Predicted class index: {predicted_class}")
    click.echo(f"Confidence: {confidence:.2%}")
    click.echo("\nAll probabilities:")
    for i, prob in enumerate(predictions[0]):
        label = class_names[i] if class_names and len(class_names) > i else f"Class {i}"
        click.echo(f"  {label}: {prob:.2%}")
    click.echo(_SEPARATOR + "\n")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
