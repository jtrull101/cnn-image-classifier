"""Model trainer for managing the training process."""

import gc
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from img_classifier_config import BaseConfig
from img_classifier_data import BaseDataLoader
from img_classifier_models import BaseModel
from img_classifier_training.callbacks import AccuracyThresholdCallback, ValTargetStop


logger = logging.getLogger(__name__)


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

    def prepare_data(self) -> tuple[np.ndarray, ...]:
        """Load and prepare all datasets.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("Loading training data...")
        X_train, y_train = self.data_loader.load_train_data()

        logger.info("Loading test data...")
        X_test, y_test = self.data_loader.load_test_data()

        # Split test data into validation and test sets. A fixed eval seed keeps this
        # partition identical across attempts so cross-attempt model selection is fair;
        # stratification keeps class balance in both halves.
        X_val, X_test, y_val, y_test = self.data_loader.split_data(
            X_test,
            y_test,
            self.config.test_split,
            seed=getattr(self.config, "eval_split_seed", None),
            stratify=getattr(self.config, "stratify_eval_split", False),
        )

        # Reduce dataset size if configured
        if self.config.data_percent < 1.0:
            logger.info("Reducing dataset to %.0f%%...", self.config.data_percent * 100)
            X_train, y_train = self.data_loader.reduce_dataset(
                X_train, y_train, self.config.data_percent
            )
            X_val, y_val = self.data_loader.reduce_dataset(X_val, y_val, self.config.data_percent)
            X_test, y_test = self.data_loader.reduce_dataset(
                X_test, y_test, self.config.data_percent
            )

        # The image loader reads BGR (cv2); pretrained backbones and the app's DJL serving
        # path both expect RGB, so flip the channel order when a backbone is in use.
        if getattr(self.config, "backbone", None):
            X_train = X_train[..., ::-1]
            X_val = X_val[..., ::-1]
            X_test = X_test[..., ::-1]

        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.config.num_classes)
        y_val = tf.keras.utils.to_categorical(y_val, self.config.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.config.num_classes)

        logger.info(
            "Data prepared: training=%s  validation=%s  test=%s",
            X_train.shape,
            X_val.shape,
            X_test.shape,
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        """Create and return list of callbacks.

        Returns:
            List of Keras callbacks
        """
        callbacks = []

        # Early stopping
        if self.config.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor=getattr(self.config, "early_stopping_monitor", "val_loss"),
                mode=getattr(self.config, "early_stopping_mode", "min"),
                patience=self.config.early_stopping_patience,
                restore_best_weights=getattr(self.config, "restore_best_weights", False),
                verbose=0,
            )
            callbacks.append(early_stopping)

        # Reduce learning rate on plateau
        if getattr(self.config, "use_reduce_lr", False):
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience,
                    min_lr=self.config.min_lr,
                    verbose=0,
                )
            )

        # Stop on a validation-accuracy target
        target_val_accuracy = getattr(self.config, "target_val_accuracy", None)
        if target_val_accuracy is not None:
            callbacks.append(ValTargetStop(target_val_accuracy))

        # Model checkpoint
        if self.config.use_model_checkpoint:
            filepath = self.config.models_dir / "optimal_weights_{val_accuracy:.0%}.keras"
            checkpoint = ModelCheckpoint(
                str(filepath),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=0,
                initial_value_threshold=0.9,
            )
            callbacks.append(checkpoint)

        # Accuracy threshold stopping
        if self.config.use_accuracy_threshold_stopping:
            acc_callback = AccuracyThresholdCallback(threshold=self.config.accuracy_threshold)
            callbacks.append(acc_callback)

        return callbacks

    def _build_augmenter(self) -> "tf.keras.Sequential":
        """Build a train-only augmentation pipeline from config knobs."""
        layers: list = []
        if getattr(self.config, "aug_horizontal_flip", False):
            layers.append(tf.keras.layers.RandomFlip("horizontal"))
        if self.config.aug_rotation:
            layers.append(tf.keras.layers.RandomRotation(self.config.aug_rotation))
        if self.config.aug_translation:
            layers.append(
                tf.keras.layers.RandomTranslation(
                    self.config.aug_translation, self.config.aug_translation
                )
            )
        if self.config.aug_zoom:
            layers.append(tf.keras.layers.RandomZoom(self.config.aug_zoom))
        return tf.keras.Sequential(layers, name="augmentation")

    @staticmethod
    def _balanced_sample_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
        """Class-balanced per-sample weights from one-hot labels (balanced data -> ~1.0)."""
        labels = np.argmax(y_train, axis=1)
        counts = np.bincount(labels, minlength=num_classes)
        n_present = int(np.count_nonzero(counts))
        per_class = np.divide(
            float(len(labels)),
            n_present * counts,
            out=np.ones(len(counts), dtype="float64"),
            where=counts > 0,
        )
        return per_class[labels].astype("float32")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        extra_callbacks: list | None = None,
    ) -> tf.keras.callbacks.History:
        """Train the model.

        When ``config.use_data_augmentation`` / ``config.use_class_weights`` are set, training
        runs over a ``tf.data`` pipeline that augments the training stream and/or applies
        class-balanced sample weights. When ``config.backbone`` is set with
        ``config.finetune_epochs > 0``, a second fine-tuning phase unfreezes the base at a
        reduced learning rate and its history is appended.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            extra_callbacks: Optional extra Keras callbacks (e.g. progress logging) appended
                to the config-driven callbacks.

        Returns:
            Training history object (covering both phases when fine-tuning).
        """
        # Compile model if not already compiled
        if self.model.model is None:
            self.model.compile()
        if self.model.model is None:
            raise RuntimeError("Model must be built before training")

        callbacks = self.get_callbacks()
        if extra_callbacks:
            callbacks = callbacks + list(extra_callbacks)

        use_aug = getattr(self.config, "use_data_augmentation", False)
        use_weights = getattr(self.config, "use_class_weights", False)

        fit_kwargs: dict = {
            "epochs": self.config.num_epochs,
            "verbose": 0,
            "validation_data": (X_val, y_val),
            "callbacks": callbacks,
        }

        if use_aug or use_weights:
            batch = self.config.batch_size
            if use_weights:
                sw = self._balanced_sample_weights(y_train, self.config.num_classes)
                ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, sw))
            else:
                ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            ds = ds.shuffle(min(len(X_train), 4096)).batch(batch)
            if use_aug:
                augment = self._build_augmenter()
                if use_weights:
                    ds = ds.map(
                        lambda x, y, w: (augment(x, training=True), y, w),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )
                else:
                    ds = ds.map(
                        lambda x, y: (augment(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )
            train_data: object = ds.prefetch(tf.data.AUTOTUNE)
            fit_args: tuple = (train_data,)
        else:
            fit_args = (X_train, y_train)
            fit_kwargs["batch_size"] = self.config.batch_size

        logger.info("Starting training...")
        start_time = time.time()
        history = self.model.model.fit(*fit_args, **fit_kwargs)

        # Optional fine-tuning phase: unfreeze the backbone at a reduced learning rate.
        finetune_epochs = getattr(self.config, "finetune_epochs", 0)
        if getattr(self.config, "backbone", None) and finetune_epochs > 0:
            self.model.model.trainable = True
            ft_lr = self.config.learning_rate / self.config.finetune_lr_divisor
            self.model.compile(learning_rate=ft_lr)
            logger.info("Fine-tuning: unfroze backbone, lr=%.2e, epochs=%d", ft_lr, finetune_epochs)
            ft_kwargs = dict(fit_kwargs)
            ft_kwargs["epochs"] = finetune_epochs
            ft_history = self.model.model.fit(*fit_args, **ft_kwargs)
            for key, values in ft_history.history.items():
                history.history.setdefault(key, [])
                history.history[key].extend(values)

        elapsed_time = time.time() - start_time
        logger.info("Training completed in %.0f seconds", elapsed_time)

        self.history = history
        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
        """Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Tuple of (loss, accuracy)
        """
        if self.model.model is None:
            raise RuntimeError("Model must be built before evaluation")

        logger.info("Evaluating model on test set...")
        results = self.model.model.evaluate(X_test, y_test, verbose=0)
        # evaluate returns list[float] or tuple when metrics are present, or float if only loss
        if isinstance(results, (list, tuple)):
            loss, acc = results[0], results[1]
        else:
            # Should not happen since we compile with metrics=['accuracy']
            loss = float(results)
            acc = 0.0
        logger.info("Test Loss: %.4f  Test Accuracy: %.4f (%.2f%%)", loss, acc, acc * 100)
        return loss, acc

    def plot_history(self, save_path: Path | None = None) -> None:
        """Plot training history.

        Args:
            save_path: Optional path to save the plot
        """
        if self.history is None:
            logger.warning("No training history available")
            return

        df = pd.DataFrame(self.history.history).rename_axis("epoch").reset_index()
        df = df.melt(id_vars=["epoch"])

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        axes_flat = np.ravel(axes)
        for ax, metric in zip(axes_flat, ["loss", "accuracy"], strict=False):
            ax.set_title(f"{metric.title()} Plot")
            df_metric = df[df["variable"].str.contains(metric)]
            sns.lineplot(data=df_metric, x="epoch", y="value", hue="variable", ax=ax)

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info("Plot saved to %s", save_path)

        plt.show()

    def save_model(
        self, acc: float, loss: float, elapsed_time: float, force_save: bool = False
    ) -> Path | None:
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
            logger.info(
                "Model accuracy %.2f%% below threshold %.2f%%, not saving",
                acc * 100,
                self.config.min_accuracy_to_save * 100,
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

    def log_results(self, acc: float, loss: float, elapsed_time: float) -> None:
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
            with log_file.open("w") as f:
                f.write(f"Training Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(
                    "timestamp,accuracy,loss,data_percent,batch_size,learning_rate,epochs,elapsed_time\n"
                )

        # Append results
        with log_file.open("a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(
                f"{timestamp},{acc:.4f},{loss:.4f},"
                f"{self.config.data_percent},{self.config.batch_size},"
                f"{self.config.learning_rate},{self.config.num_epochs},"
                f"{int(elapsed_time)}\n"
            )

        logger.info("Results logged to %s", log_file)

    def run(self, plot: bool = False, force_save: bool = False) -> tuple[float, float]:
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

    def cleanup(self) -> None:
        """Clean up resources after training."""
        if self.model.model:
            del self.model.model
            self.model.model = None

        K.clear_session()
        gc.collect()
        logger.info("Cleanup completed")
