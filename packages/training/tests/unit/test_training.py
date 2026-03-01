"""Unit tests for training modules using mocking for fast execution.

This file contains pure unit tests that use mocking to verify the training
module's behavior without actually running TensorFlow training loops or
loading real data. These tests should execute quickly (<1s each).
"""

from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest


pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestAccuracyThresholdCallback:
    """Unit tests for AccuracyThresholdCallback using mocks."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures with mocked dependencies."""
        # Mock the callback to avoid TensorFlow import
        with patch("img_classifier_training.callbacks.tf"):
            from img_classifier_training.callbacks import AccuracyThresholdCallback

            self.callback_class = AccuracyThresholdCallback

        yield

    def test_initialization(self):
        """Test callback initialization with custom threshold."""
        callback = self.callback_class(threshold=0.99)
        assert callback.threshold == 0.99

    def test_default_threshold(self):
        """Test default threshold value."""
        callback = self.callback_class()
        assert callback.threshold == 0.995

    def test_stops_on_high_accuracy(self):
        """Test that callback stops training when accuracy exceeds threshold."""
        callback = self.callback_class(threshold=0.95)
        mock_model = MagicMock()
        mock_model.stop_training = False
        callback.set_model(mock_model)

        logs = {"accuracy": 0.96, "val_accuracy": 0.94}
        callback.on_epoch_end(0, logs)

        assert callback.model.stop_training is True

    def test_stops_on_high_val_accuracy(self):
        """Test that callback stops training when validation accuracy exceeds threshold."""
        callback = self.callback_class(threshold=0.95)
        mock_model = MagicMock()
        mock_model.stop_training = False
        callback.set_model(mock_model)

        logs = {"accuracy": 0.90, "val_accuracy": 0.96}
        callback.on_epoch_end(0, logs)

        assert callback.model.stop_training is True

    def test_does_not_stop_on_low_accuracy(self):
        """Test that callback doesn't stop when accuracy is below threshold."""
        callback = self.callback_class(threshold=0.95)
        mock_model = MagicMock()
        mock_model.stop_training = False
        callback.set_model(mock_model)

        logs = {"accuracy": 0.90, "val_accuracy": 0.85}
        callback.on_epoch_end(0, logs)

        assert callback.model.stop_training is False

    def test_handles_none_logs(self):
        """Test that callback handles None logs gracefully."""
        callback = self.callback_class()
        mock_model = MagicMock()
        callback.set_model(mock_model)

        # Should not raise an error
        callback.on_epoch_end(0, None)

    def test_handles_missing_accuracy_keys(self):
        """Test that callback handles logs without accuracy keys."""
        callback = self.callback_class(threshold=0.95)
        mock_model = MagicMock()
        mock_model.stop_training = False
        callback.set_model(mock_model)

        logs = {"loss": 0.5}
        callback.on_epoch_end(0, logs)

        # Should not stop if accuracy keys are missing
        assert callback.model.stop_training is False


@pytest.mark.unit
class TestTrainer:
    """Unit tests for Trainer class using mocked dependencies."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures with mocked dependencies."""
        self.temp_dir = isolated_tmp_dir

        # Mock the config
        self.mock_config = Mock()
        self.mock_config.working_dir = self.temp_dir
        self.mock_config.num_classes = 4
        self.mock_config.input_shape = (128, 128, 3)
        self.mock_config.batch_size = 32
        self.mock_config.num_epochs = 10
        self.mock_config.validation_split = 0.2
        self.mock_config.data_percent = 1.0
        self.mock_config.use_early_stopping = True
        self.mock_config.use_model_checkpoint = True
        self.mock_config.use_accuracy_threshold_stopping = True
        self.mock_config.project_name = "test_project"
        self.mock_config.learning_rate = 0.001
        self.mock_config.min_accuracy_to_save = 0.0
        self.mock_config.test_split = 0.5
        self.mock_config.logs_dir = self.temp_dir / "logs"
        self.mock_config.models_dir = self.temp_dir / "models"

        # Mock the model
        self.mock_model = Mock()
        self.mock_keras_model = Mock()
        self.mock_model.model = self.mock_keras_model
        self.mock_model.save = Mock()
        self.mock_model.compile = Mock()

        # Mock the data loader
        self.mock_loader = Mock()

        # Import and create trainer with mocked dependencies
        with patch("img_classifier_training.trainer.BaseConfig"):
            with patch("img_classifier_training.trainer.BaseModel"):
                with patch("img_classifier_training.trainer.BaseDataLoader"):
                    from img_classifier_training import Trainer

                    self.trainer = Trainer(self.mock_config, self.mock_model, self.mock_loader)

        yield

    def test_initialization(self):
        """Test Trainer initialization with mocked dependencies."""
        assert self.trainer.config == self.mock_config
        assert self.trainer.model == self.mock_model
        assert self.trainer.data_loader == self.mock_loader
        assert self.trainer.history is None

    def test_prepare_data_calls_loader(self):
        """Test that prepare_data calls data loader methods."""
        # Setup mock data
        X_train = np.random.rand(100, 128, 128, 3)
        y_train = np.random.randint(0, 4, 100)
        X_test = np.random.rand(40, 128, 128, 3)
        y_test = np.random.randint(0, 4, 40)

        self.mock_loader.load_train_data.return_value = (X_train, y_train)
        self.mock_loader.load_test_data.return_value = (X_test, y_test)
        self.mock_loader.split_data.return_value = (
            X_test[:20],
            X_test[20:],
            y_test[:20],
            y_test[20:],
        )

        # Call prepare_data
        result = self.trainer.prepare_data()

        # Verify loader methods were called
        self.mock_loader.load_train_data.assert_called_once()
        self.mock_loader.load_test_data.assert_called_once()
        assert len(result) == 6

    def test_prepare_data_with_reduction(self):
        """Test data preparation respects data_percent config."""
        self.mock_config.data_percent = 0.5

        X_train = np.random.rand(100, 128, 128, 3)
        y_train = np.random.randint(0, 4, 100)
        X_test = np.random.rand(40, 128, 128, 3)
        y_test = np.random.randint(0, 4, 40)

        self.mock_loader.load_train_data.return_value = (X_train, y_train)
        self.mock_loader.load_test_data.return_value = (X_test, y_test)
        self.mock_loader.split_data.return_value = (
            X_test[:20],
            X_test[20:],
            y_test[:20],
            y_test[20:],
        )
        self.mock_loader.reduce_dataset.side_effect = [
            (X_train[:50], y_train[:50]),
            (X_test[:10], y_test[:10]),
            (X_test[20:30], y_test[20:30]),
        ]

        self.trainer.prepare_data()

        # Verify reduce_dataset was called when data_percent < 1.0
        assert self.mock_loader.reduce_dataset.call_count == 3

    def test_get_callbacks_returns_list(self):
        """Test that get_callbacks returns a list."""
        callbacks = self.trainer.get_callbacks()
        assert isinstance(callbacks, list)

    def test_get_callbacks_respects_config(self):
        """Test that callbacks respect configuration flags."""
        self.mock_config.use_early_stopping = False
        self.mock_config.use_model_checkpoint = False
        self.mock_config.use_accuracy_threshold_stopping = False

        callbacks = self.trainer.get_callbacks()

        # Should return empty list when all callbacks disabled
        assert len(callbacks) == 0

    def test_train_calls_model_fit(self):
        """Test that train method calls model.fit."""
        X_train = np.random.rand(100, 128, 128, 3)
        y_train = np.random.rand(100, 4)
        X_val = np.random.rand(20, 128, 128, 3)
        y_val = np.random.rand(20, 4)

        # Mock history object
        mock_history = Mock()
        mock_history.history = {"loss": [0.5], "val_loss": [0.6]}
        self.mock_keras_model.fit.return_value = mock_history

        history = self.trainer.train(X_train, y_train, X_val, y_val)

        # Verify model.fit was called
        self.mock_keras_model.fit.assert_called_once()
        assert history == mock_history

    def test_evaluate_calls_model_evaluate(self):
        """Test that evaluate method calls model.evaluate."""
        X_test = np.random.rand(20, 128, 128, 3)
        y_test = np.random.rand(20, 4)

        self.mock_keras_model.evaluate.return_value = [0.3, 0.92]

        loss, acc = self.trainer.evaluate(X_test, y_test)

        # Verify model.evaluate was called
        self.mock_keras_model.evaluate.assert_called_once()
        assert loss == 0.3
        assert acc == 0.92

    def test_save_model_calls_model_save(self):
        """Test that save_model delegates to model.save."""
        path = self.trainer.save_model(acc=0.95, loss=0.3, elapsed_time=100.0, force_save=True)

        assert path is not None
        # Verify model.save was called with the generated path
        self.mock_model.save.assert_called_once()
        saved_path = self.mock_model.save.call_args[0][0]
        assert saved_path.parent == self.mock_config.models_dir
