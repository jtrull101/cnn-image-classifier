"""
Example: Using the Generalized Image Classification System

This script demonstrates how to use the new modularized system
to train models on any image classification dataset.
"""

from pathlib import Path

from img_classifier_config import DatasetConfig, DatasetDetector
from img_classifier_training import HyperparameterSpace, TrainingOrchestrator


def example_1_simple_training():
    """Example 1: Simple training with auto-detection."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Training with Auto-Detection")
    print("=" * 60 + "\n")

    # Point to your dataset directory
    dataset_path = Path("/path/to/your/dataset")

    # Create orchestrator with auto-detection
    orchestrator = TrainingOrchestrator.from_dataset_path(
        dataset_path=dataset_path,
        project_name="my_classifier",
        working_dir=Path("/tmp/my_classifier/"),
    )

    # Run training
    _model = orchestrator.run(plot=True)

    print(f"Training completed! Model saved in {orchestrator.config.models_dir}")


def example_2_custom_config():
    """Example 2: Training with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Training with Custom Configuration")
    print("=" * 60 + "\n")

    # Create custom configuration
    config = DatasetConfig(
        project_name="custom_classifier",
        working_dir=Path("/tmp/custom_classifier/"),
        dataset_name="my_dataset",
        data_path=Path("/path/to/your/dataset"),
        # Image settings
        image_size=(128, 128),
        color_channels=3,
        # Training parameters
        batch_size=32,
        num_epochs=50,
        learning_rate=0.001,
        # Model architecture
        architecture_complexity="medium",
        dropout_rate=0.3,
        # Use less data for faster experimentation
        data_percent=0.1,  # Use only 10% of data
    )

    # Create orchestrator with custom config
    orchestrator = TrainingOrchestrator(config)

    # Run training
    _model = orchestrator.run(plot=True)


def example_3_hyperparameter_optimization():
    """Example 3: Hyperparameter optimization."""
    print("\n" + "=" * 60)
    print("Example 3: Hyperparameter Optimization")
    print("=" * 60 + "\n")

    dataset_path = Path("/path/to/your/dataset")

    # Create custom search space
    search_space = HyperparameterSpace(
        learning_rates=[0.0001, 0.001, 0.01],
        batch_sizes=[16, 32, 64],
        dropout_rates=[0.3, 0.4, 0.5],
        architectures=["simple", "medium"],
        num_epochs=[20, 30],
    )

    # Create orchestrator with optimization enabled
    orchestrator = TrainingOrchestrator.from_dataset_path(
        dataset_path=dataset_path,
        project_name="optimized_classifier",
        optimize_hyperparameters=True,
        optimizer_type="random",  # Options: 'random', 'grid', 'bayesian'
        search_space=search_space,
        max_trials=20,
    )

    # Run optimization
    _model = orchestrator.run()

    # Print results
    if orchestrator.optimizer:
        print(orchestrator.optimizer.get_summary())

        # Get best hyperparameters
        best = orchestrator.optimizer.best_result
        print(f"\nBest validation accuracy: {best.val_accuracy:.4f}")
        print(f"Best test accuracy: {best.test_accuracy:.4f}")
        print("\nBest hyperparameters:")
        for key, value in best.hyperparameters.items():
            print(f"  {key}: {value}")


def example_4_dataset_detection():
    """Example 4: Dataset detection and analysis."""
    print("\n" + "=" * 60)
    print("Example 4: Dataset Detection and Analysis")
    print("=" * 60 + "\n")

    dataset_path = Path("/path/to/your/dataset")

    # Create detector
    detector = DatasetDetector(dataset_path)

    # Detect dataset characteristics
    info = detector.detect()

    # Print information
    print(f"Dataset: {info['dataset_path']}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Class names: {', '.join(info['class_names'])}")
    print(f"Total images: {info['total_images']}")
    print("\nClass distribution:")
    for cls, count in info["class_distribution"].items():
        print(f"  {cls}: {count}")
    print(f"\nBalanced: {info['is_balanced']}")
    print(f"Has train/test split: {info['has_train_test_split']}")
    print(f"Recommended complexity: {info['recommended_complexity']}")

    if info["sample_image_shape"]:
        h, w, c = info["sample_image_shape"]
        print(f"Sample image shape: {h}x{w}x{c}")

    # Create configuration from detected info
    config = detector.create_config(
        project_name="detected_classifier",
        working_dir=Path("/tmp/detected_classifier/"),
    )

    # Save configuration to YAML
    config.to_yaml(Path("dataset_config.yaml"))
    print("\nConfiguration saved to dataset_config.yaml")


def example_5_manual_architecture():
    """Example 5: Creating custom architecture."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Architecture")
    print("=" * 60 + "\n")

    from img_classifier_models import ArchitectureFactory, ArchitectureSpec

    # Create configuration
    config = DatasetConfig(
        project_name="custom_arch",
        num_classes=5,
        image_size=(128, 128),
        color_channels=3,
    )

    # Define custom architecture
    custom_spec = ArchitectureSpec(
        name="MyCustomCNN",
        conv_blocks=[
            {"filters": 32, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
            {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
            {"filters": 128, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
            {"filters": 256, "kernel_size": (3, 3), "activation": "relu", "pooling": True},
        ],
        dense_layers=[512, 256],
        use_batch_norm=True,
        use_global_pooling=True,
        dropout_rate=0.4,
    )

    # Build model
    model = ArchitectureFactory.create(config, custom_spec=custom_spec)

    # Display architecture
    print("\nModel architecture:")
    model.summary()


def example_6_using_yaml_config():
    """Example 6: Loading configuration from YAML file."""
    print("\n" + "=" * 60)
    print("Example 6: Loading Configuration from YAML")
    print("=" * 60 + "\n")

    # Create a sample config file first
    config = DatasetConfig(
        project_name="yaml_example",
        working_dir=Path("/tmp/yaml_example/"),
        dataset_name="my_dataset",
        data_path=Path("/path/to/your/dataset"),
        num_classes=4,
        image_size=(128, 128),
        batch_size=32,
        num_epochs=25,
        learning_rate=0.001,
        architecture_complexity="medium",
    )

    config_path = Path("my_config.yaml")
    config.to_yaml(config_path)
    print(f"Configuration saved to {config_path}")

    # Load configuration from YAML
    loaded_config = DatasetConfig.from_yaml(config_path)

    print("\nLoaded configuration:")
    print(f"  Project: {loaded_config.project_name}")
    print(f"  Classes: {loaded_config.num_classes}")
    print(f"  Image size: {loaded_config.image_size}")
    print(f"  Batch size: {loaded_config.batch_size}")
    print(f"  Epochs: {loaded_config.num_epochs}")

    # Use loaded config for training
    _orchestrator = TrainingOrchestrator(loaded_config)
    # _model = _orchestrator.run()


def example_7_quick_experiment():
    """Example 7: Quick experiment with small dataset."""
    print("\n" + "=" * 60)
    print("Example 7: Quick Experiment")
    print("=" * 60 + "\n")

    dataset_path = Path("/path/to/your/dataset")

    # Quick search space
    search_space = HyperparameterSpace.quick()

    # Detect dataset
    detector = DatasetDetector(dataset_path)
    config = detector.create_config(
        project_name="quick_experiment",
        # Use small subset for speed
        data_percent=0.1,
        # Fewer epochs
        num_epochs=10,
        # Simple architecture
        architecture_complexity="simple",
    )

    # Run quick optimization
    orchestrator = TrainingOrchestrator(
        config,
        optimize_hyperparameters=True,
        optimizer_type="random",
        search_space=search_space,
        max_trials=5,  # Just 5 trials
    )

    _model = orchestrator.run()

    print("\nQuick experiment completed!")
    if orchestrator.optimizer:
        print(orchestrator.optimizer.get_summary())


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     Generalized Image Classification System Examples      ║
    ╚════════════════════════════════════════════════════════════╝

    Choose an example to run:

    1. Simple training with auto-detection
    2. Training with custom configuration
    3. Hyperparameter optimization
    4. Dataset detection and analysis
    5. Creating custom architecture
    6. Loading configuration from YAML
    7. Quick experiment with small dataset

    Note: Update the dataset paths in the examples before running!
    """)

    # Uncomment the example you want to run:
    # example_1_simple_training()
    # example_2_custom_config()
    # example_3_hyperparameter_optimization()
    # example_4_dataset_detection()
    # example_5_manual_architecture()
    # example_6_using_yaml_config()
    # example_7_quick_experiment()
