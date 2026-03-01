"""Custom callbacks for model training."""

import tensorflow as tf


class AccuracyThresholdCallback(tf.keras.callbacks.Callback):
    """Stop training when accuracy threshold is reached.

    This callback helps prevent overfitting by stopping training
    when the model achieves very high accuracy on the training set.
    """

    def __init__(self, threshold: float = 0.995):
        """Initialize the callback.

        Args:
            threshold: Accuracy threshold to stop at
        """
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Check accuracy at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        if logs is None:
            return

        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)

        if acc >= self.threshold or val_acc >= self.threshold:
            print(
                f"\nReached {self.threshold * 100}% accuracy. Stopping training to prevent overfitting."
            )
            self.model.stop_training = True
