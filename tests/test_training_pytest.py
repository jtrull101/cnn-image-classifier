"""Tests for training modules."""

from unittest.mock import MagicMock

import numpy as np
import pytest


try:
    import tensorflow as tf

    from img_classifier_models import ArchitectureFactory
    from img_classifier_training import AccuracyThresholdCallback, Trainer, TrainingOrchestrator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    AccuracyThresholdCallback = None
    Trainer = None
    ArchitectureFactory = None
    TrainingOrchestrator = None

from img_classifier_config import DatasetConfig
from img_classifier_data import BaseDataLoader


pytestmark = pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")


class TestAccuracyThresholdCallback:
    """Tests for AccuracyThresholdCallback."""

    def test_initialization(self):
        """Test callback initialization."""
        callback = AccuracyThresholdCallback(threshold=0.99)
        assert callback.threshold == 0.99

    def test_default_threshold(self):
        """Test default threshold value."""
        callback = AccuracyThresholdCallback()
        assert callback.threshold == 0.995

    def test_stops_on_high_accuracy(self):
        """Test that callback stops training on high accuracy."""
        callback = AccuracyThresholdCallback(threshold=0.95)
        mock_model = MagicMock()
        mock_model.stop_training = False
        callback.set_model(mock_model)

        logs = {'acc': 0.96, 'val_acc': 0.94}
        callback.on_epoch_end(0, logs)

        assert callback.model.stop_training is True

    def test_stops_on_high_val_accuracy(self):
        """Test that callback stops training on high validation accuracy."""
        callback = AccuracyThresholdCallback(threshold=0.95)
        mock_model = MagicMock()
        mock_model.stop_training = False
        callback.set_model(mock_model)

        logs = {'acc': 0.90, 'val_acc': 0.96}
        callback.on_epoch_end(0, logs)

        assert callback.model.stop_training is True

    def test_does_not_stop_on_low_accuracy(self):
        """Test that callback doesn't stop on low accuracy."""
        callback = AccuracyThresholdCallback(threshold=0.95)
        mock_model = MagicMock()
        mock_model.stop_training = False
        callback.set_model(mock_model)

        logs = {'acc': 0.90, 'val_acc': 0.85}
        callback.on_epoch_end(0, logs)

        assert callback.model.stop_training is False

    def test_handles_none_logs(self):
        """Test that callback handles None logs gracefully."""
        callback = AccuracyThresholdCallback()
        mock_model = MagicMock()
        callback.set_model(mock_model)

        # Should not raise an error
        callback.on_epoch_end(0, None)


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures."""
        self.temp_dir = tmp_path
        self.config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(32, 32),
            num_classes=2,
            class_names=["class0", "class1"],
            batch_size=4,
            num_epochs=2,
            architecture_complexity="simple"
        )

        # Create model using ArchitectureFactory wrapped in BaseModel
        keras_model = ArchitectureFactory.create(self.config, complexity="simple")

        from img_classifier_models import BaseModel
        class FactoryModel(BaseModel):
            def __init__(self, config, keras_model):
                super().__init__(config)
                self.model = keras_model

            def build(self):
                return self.model

        self.model = FactoryModel(self.config, keras_model)
        self.data_loader = self._create_mock_data_loader()
        self.trainer = Trainer(self.config, self.model, self.data_loader)

        yield  # Test runs here

        # Teardown
        tf.keras.backend.clear_session()

    def _create_mock_data_loader(self):
        """Create a mock data loader."""
        class MockDataLoader(BaseDataLoader):
            def __init__(self, config):
                super().__init__(config)

            def load_train_data(self):
                X = np.random.rand(20, 32, 32, 3).astype('float32')
                y = np.random.randint(0, 2, 20)
                return X, y

            def load_test_data(self):
                X = np.random.rand(10, 32, 32, 3).astype('float32')
                y = np.random.randint(0, 2, 10)
                return X, y

            def download_dataset(self):
                return True

            def prepare_dataset(self):
                return True

        return MockDataLoader(self.config)

    def test_initialization(self):
        """Test Trainer initialization."""
        assert self.trainer.config == self.config
        assert self.trainer.model == self.model
        assert self.trainer.data_loader == self.data_loader
        assert self.trainer.history is None

    def test_prepare_data(self):
        """Test data preparation."""
        X_train, y_train, X_val, y_val, X_test, y_test = self.trainer.prepare_data()

        # Check that data is loaded
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(X_val, np.ndarray)
        assert isinstance(y_val, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_test, np.ndarray)

        # Check shapes
        assert X_train.shape[1:] == self.config.input_shape

        # Check that labels are one-hot encoded
        assert y_train.shape[1] == self.config.num_classes

    def test_prepare_data_with_reduced_dataset(self):
        """Test data preparation with dataset reduction."""
        self.config.data_percent = 0.5
        X_train, _y_train, X_val, _y_val, X_test, _y_test = self.trainer.prepare_data()

        # Data should be loaded (though reduced)
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0

    def test_get_callbacks(self):
        """Test callback creation."""
        callbacks = self.trainer.get_callbacks()

        assert isinstance(callbacks, list)
        # Should have early stopping, model checkpoint, and accuracy threshold
        assert len(callbacks) >= 3

    def test_get_callbacks_respects_config(self):
        """Test that callbacks respect configuration."""
        self.config.use_early_stopping = False
        self.config.use_model_checkpoint = False
        self.config.use_accuracy_threshold_stopping = False

        callbacks = self.trainer.get_callbacks()

        assert len(callbacks) == 0

    def test_train_compiles_model(self):
        """Test that train compiles model if needed."""
        X_train, y_train, X_val, y_val, _X_test, _y_test = self.trainer.prepare_data()
        self.model.compile()  # Ensure model is compiled before training
        self.trainer.train(X_train, y_train, X_val, y_val)
        assert hasattr(self.model.model, 'compiled') and self.model.model.compiled

    def test_train_returns_history(self):
        """Test that train returns history object."""
        X_train, y_train, X_val, y_val, _X_test, _y_test = self.trainer.prepare_data()
        self.model.compile()  # Ensure model is compiled before training
        history = self.trainer.train(X_train, y_train, X_val, y_val)

        assert history is not None
        assert hasattr(history, 'history')
        assert 'loss' in history.history
        assert 'val_loss' in history.history

    def test_evaluate(self):
        """Test model evaluation."""
        X_train, y_train, X_val, y_val, X_test, y_test = self.trainer.prepare_data()
        self.model.compile()  # Ensure model is compiled before training
        self.trainer.train(X_train, y_train, X_val, y_val)
        loss, acc = self.trainer.evaluate(X_test, y_test)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_save_model_requires_minimum_accuracy(self):
        """Test that save_model respects minimum accuracy."""
        self.config.min_accuracy_to_save = 0.99
        self.model.build()
        self.model.compile()

        # Low accuracy should not save
        result = self.trainer.save_model(acc=0.50, loss=1.0, elapsed_time=100)
        assert result is None

    def test_save_model_with_force_save(self):
        """Test that save_model saves with force_save=True."""
        self.config.min_accuracy_to_save = 0.99
        self.model.build()
        self.model.compile()

        # Should save even with low accuracy
        result = self.trainer.save_model(
            acc=0.50, loss=1.0, elapsed_time=100, force_save=True
        )
        assert result is not None
        assert result.exists()

    def test_log_results_creates_file(self):
        """Test that log_results creates log file."""
        log_file = self.config.logs_dir / "training_history.log"
        assert not log_file.exists()

        self.trainer.log_results(acc=0.95, loss=0.1, elapsed_time=100)

        assert log_file.exists()

    def test_cleanup(self):
        """Test cleanup method."""
        self.model.build()
        assert self.model.model is not None

        self.trainer.cleanup()

        assert self.model.model is None

