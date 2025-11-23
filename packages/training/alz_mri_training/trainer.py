"""Model trainer for managing the training process."""

import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

from alz_mri_config import BaseConfig
from alz_mri_data import BaseDataLoader
from alz_mri_models import BaseModel
from .callbacks import AccuracyThresholdCallback


class Trainer:
    """Handles model training and evaluation.

    This class manages the complete training pipeline including:
    - Data loading and preprocessing
    - Model compilation
    - Training with callbacks
    - Evaluation and logging
    - Model saving
    """

    def __init__(self, config: BaseConfig, model: BaseModel, data_loader: BaseDataLoader):
        """Initialize the trainer.

        Args:
            config: Configuration object
            model: Model to train
            data_loader: Data loader for loading datasets
        """
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.history = None

    def prepare_data(self) -> Tuple[np.ndarray, ...]:
        """Load and prepare all datasets.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print("Loading training data...")
        X_train, y_train = self.data_loader.load_train_data()

        print("Loading test data...")
        X_test, y_test = self.data_loader.load_test_data()

        # Split test data into validation and test sets
        X_val, X_test, y_val, y_test = self.data_loader.split_data(
            X_test, y_test, self.config.test_split
        )

        # Reduce dataset size if configured
        if self.config.data_percent < 1.0:
            print(f"Reducing dataset to {self.config.data_percent * 100}%...")
            X_train, y_train = self.data_loader.reduce_dataset(
                X_train, y_train, self.config.data_percent
            )
            X_val, y_val = self.data_loader.reduce_dataset(X_val, y_val, self.config.data_percent)
            X_test, y_test = self.data_loader.reduce_dataset(
                X_test, y_test, self.config.data_percent
            )

        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.config.num_classes)
        y_val = tf.keras.utils.to_categorical(y_val, self.config.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.config.num_classes)

        print("Data prepared:")
        print(f"  Training: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Create and return list of callbacks.

        Returns:
            List of Keras callbacks
        """
        callbacks = []

        # Early stopping
        if self.config.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=self.config.early_stopping_patience,
                verbose=1,
            )
            callbacks.append(early_stopping)

        # Model checkpoint
        if self.config.use_model_checkpoint:
            filepath = self.config.models_dir / "optimal_weights_{val_acc:.0%}.keras"
            checkpoint = ModelCheckpoint(
                str(filepath),
                monitor="val_acc",
                mode="max",
                save_best_only=True,
                verbose=1,
                initial_value_threshold=0.9,
            )
            callbacks.append(checkpoint)

        # Accuracy threshold stopping
        if self.config.use_accuracy_threshold_stopping:
            acc_callback = AccuracyThresholdCallback(threshold=self.config.accuracy_threshold)
            callbacks.append(acc_callback)

        return callbacks

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> tf.keras.callbacks.History:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training history object
        """
        # Compile model if not already compiled
        if self.model.model is None:
            self.model.compile()

        # Get callbacks
        callbacks = self.get_callbacks()

        # Train the model
        print("\nStarting training...")
        start_time = time.time()

        assert self.model.model is not None, "Model must be built before training"
        history = self.model.model.fit(
            X_train,
            y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.num_epochs,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )

        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.0f} seconds")

        self.history = history
        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Tuple of (loss, accuracy)
        """
        assert self.model.model is not None, "Model must be built before evaluation"

        print("\nEvaluating model on test set...")
        loss, acc = self.model.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        return loss, acc

    def plot_history(self, save_path: Optional[Path] = None):
        """Plot training history.

        Args:
            save_path: Optional path to save the plot
        """
        if self.history is None:
            print("No training history available")
            return

        df = pd.DataFrame(self.history.history).rename_axis("epoch").reset_index()
        df = df.melt(id_vars=["epoch"])

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        for ax, metric in zip(axes.flat, ["loss", "acc"]):
            ax.set_title(f"{metric.title()} Plot")
            df_metric = df[df["variable"].str.contains(metric)]
            sns.lineplot(data=df_metric, x="epoch", y="value", hue="variable", ax=ax)  # type: ignore[arg-type]

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()

    def save_model(
        self, acc: float, loss: float, elapsed_time: float, force_save: bool = False
    ) -> Optional[Path]:
        """Save the trained model.

        Args:
            acc: Test accuracy
            loss: Test loss
            elapsed_time: Training time in seconds
            force_save: Force saving regardless of accuracy

        Returns:
            Path where model was saved, or None if not saved
        """
        if not force_save and acc < self.config.min_accuracy_to_save:
            print(
                f"Model accuracy {acc:.2%} below threshold {self.config.min_accuracy_to_save:.2%}, not saving"
            )
            return None

        acc_pct = f"{int(acc * 100)}%"
        data_pct = f"{int(self.config.data_percent * 100)}%"

        filename = (
            f"{self.config.project_name}_{acc_pct}_acc_"
            f"{self.config.num_epochs}_ep_{self.config.batch_size}_bs_"
            f"{self.config.learning_rate}_lr_{data_pct}_data_"
            f"{loss:.2f}_loss_{int(elapsed_time)}s.keras"
        )

        filepath = self.config.models_dir / filename
        self.model.save(filepath)

        return filepath

    def log_results(self, acc: float, loss: float, elapsed_time: float):
        """Log training results to file.

        Args:
            acc: Test accuracy
            loss: Test loss
            elapsed_time: Training time in seconds
        """
        log_file = self.config.logs_dir / "training_history.log"

        # Ensure logs directory exists
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        # Create header if file doesn't exist
        if not log_file.exists():
            with open(log_file, "w") as f:
                f.write(f"Training Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(
                    "timestamp,accuracy,loss,data_percent,batch_size,learning_rate,epochs,elapsed_time\n"
                )

        # Append results
        with open(log_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(
                f"{timestamp},{acc:.4f},{loss:.4f},"
                f"{self.config.data_percent},{self.config.batch_size},"
                f"{self.config.learning_rate},{self.config.num_epochs},"
                f"{int(elapsed_time)}\n"
            )

        print(f"Results logged to {log_file}")

    def run(self, plot: bool = False, force_save: bool = False) -> Tuple[float, float]:
        """Run the complete training pipeline.

        Args:
            plot: Whether to plot training history
            force_save: Force saving model regardless of accuracy

        Returns:
            Tuple of (test_accuracy, test_loss)
        """
        try:
            # Prepare data
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data()

            # Train model
            start_time = time.time()
            self.train(X_train, y_train, X_val, y_val)
            elapsed_time = time.time() - start_time

            # Plot training history
            if plot:
                plot_path = (
                    self.config.logs_dir
                    / f"training_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                self.plot_history(save_path=plot_path)

            # Evaluate on test set
            loss, acc = self.evaluate(X_test, y_test)

            # Save model
            self.save_model(acc, loss, elapsed_time, force_save=force_save)

            # Log results
            self.log_results(acc, loss, elapsed_time)

            return acc, loss

        finally:
            # Clean up
            self.cleanup()

    def cleanup(self):
        """Clean up resources after training."""
        if self.model.model:
            del self.model.model
            self.model.model = None

        K.clear_session()
        gc.collect()
        print("Cleanup completed")
