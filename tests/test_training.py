"""Tests for training modules."""

import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
from unittest.mock import MagicMock, patch
import tensorflow as tf

from src.alz_mri_cnn.config import BaseConfig
from src.alz_mri_cnn.training import AccuracyThresholdCallback, Trainer
from src.alz_mri_cnn.models import CNNClassifier
from src.alz_mri_cnn.data import BaseDataLoader


class TestAccuracyThresholdCallback(unittest.TestCase):
    """Tests for AccuracyThresholdCallback."""

    def test_initialization(self):
        """Test callback initialization."""
        callback = AccuracyThresholdCallback(threshold=0.99)
        self.assertEqual(callback.threshold, 0.99)

    def test_default_threshold(self):
        """Test default threshold value."""
        callback = AccuracyThresholdCallback()
        self.assertEqual(callback.threshold, 0.995)

    def test_stops_on_high_accuracy(self):
        """Test that callback stops training on high accuracy."""
        callback = AccuracyThresholdCallback(threshold=0.95)
        callback.model = MagicMock()
        callback.model.stop_training = False
        
        logs = {'acc': 0.96, 'val_acc': 0.94}
        callback.on_epoch_end(0, logs)
        
        self.assertTrue(callback.model.stop_training)

    def test_stops_on_high_val_accuracy(self):
        """Test that callback stops training on high validation accuracy."""
        callback = AccuracyThresholdCallback(threshold=0.95)
        callback.model = MagicMock()
        callback.model.stop_training = False
        
        logs = {'acc': 0.90, 'val_acc': 0.96}
        callback.on_epoch_end(0, logs)
        
        self.assertTrue(callback.model.stop_training)

    def test_does_not_stop_on_low_accuracy(self):
        """Test that callback doesn't stop on low accuracy."""
        callback = AccuracyThresholdCallback(threshold=0.95)
        callback.model = MagicMock()
        callback.model.stop_training = False
        
        logs = {'acc': 0.90, 'val_acc': 0.85}
        callback.on_epoch_end(0, logs)
        
        self.assertFalse(callback.model.stop_training)

    def test_handles_none_logs(self):
        """Test that callback handles None logs gracefully."""
        callback = AccuracyThresholdCallback()
        callback.model = MagicMock()
        
        # Should not raise an error
        callback.on_epoch_end(0, None)


class TestTrainer(unittest.TestCase):
    """Tests for Trainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = BaseConfig(
            working_dir=self.temp_dir,
            image_size=(32, 32),
            num_classes=2,
            batch_size=4,
            num_epochs=2
        )
        self.model = CNNClassifier(self.config)
        self.data_loader = self._create_mock_data_loader()
        self.trainer = Trainer(self.config, self.model, self.data_loader)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
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
        self.assertEqual(self.trainer.config, self.config)
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.data_loader, self.data_loader)
        self.assertIsNone(self.trainer.history)

    def test_prepare_data(self):
        """Test data preparation."""
        X_train, y_train, X_val, y_val, X_test, y_test = self.trainer.prepare_data()
        
        # Check that data is loaded
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(X_val, np.ndarray)
        self.assertIsInstance(y_val, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        
        # Check shapes
        self.assertEqual(X_train.shape[1:], self.config.input_shape)
        
        # Check that labels are one-hot encoded
        self.assertEqual(y_train.shape[1], self.config.num_classes)

    def test_prepare_data_with_reduced_dataset(self):
        """Test data preparation with dataset reduction."""
        self.config.data_percent = 0.5
        X_train, y_train, X_val, y_val, X_test, y_test = self.trainer.prepare_data()
        
        # Data should be loaded (though reduced)
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(X_test), 0)

    def test_get_callbacks(self):
        """Test callback creation."""
        callbacks = self.trainer.get_callbacks()
        
        self.assertIsInstance(callbacks, list)
        # Should have early stopping, model checkpoint, and accuracy threshold
        self.assertGreaterEqual(len(callbacks), 3)

    def test_get_callbacks_respects_config(self):
        """Test that callbacks respect configuration."""
        self.config.use_early_stopping = False
        self.config.use_model_checkpoint = False
        self.config.use_accuracy_threshold_stopping = False
        
        callbacks = self.trainer.get_callbacks()
        
        self.assertEqual(len(callbacks), 0)

    def test_train_compiles_model(self):
        """Test that train compiles model if needed."""
        X_train, y_train, X_val, y_val, X_test, y_test = self.trainer.prepare_data()
        
        self.assertIsNone(self.model.model)
        self.trainer.train(X_train, y_train, X_val, y_val)
        self.assertIsNotNone(self.model.model)

    def test_train_returns_history(self):
        """Test that train returns history object."""
        X_train, y_train, X_val, y_val, X_test, y_test = self.trainer.prepare_data()
        
        history = self.trainer.train(X_train, y_train, X_val, y_val)
        
        self.assertIsNotNone(history)
        self.assertTrue(hasattr(history, 'history'))
        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)

    def test_evaluate(self):
        """Test model evaluation."""
        X_train, y_train, X_val, y_val, X_test, y_test = self.trainer.prepare_data()
        
        self.trainer.train(X_train, y_train, X_val, y_val)
        loss, acc = self.trainer.evaluate(X_test, y_test)
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_save_model_requires_minimum_accuracy(self):
        """Test that save_model respects minimum accuracy."""
        self.config.min_accuracy_to_save = 0.99
        self.model.build()
        self.model.compile()
        
        # Low accuracy should not save
        result = self.trainer.save_model(acc=0.50, loss=1.0, elapsed_time=100)
        self.assertIsNone(result)

    def test_save_model_with_force_save(self):
        """Test that save_model saves with force_save=True."""
        self.config.min_accuracy_to_save = 0.99
        self.model.build()
        self.model.compile()
        
        # Should save even with low accuracy
        result = self.trainer.save_model(
            acc=0.50, loss=1.0, elapsed_time=100, force_save=True
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.exists())

    def test_log_results_creates_file(self):
        """Test that log_results creates log file."""
        log_file = self.config.logs_dir / "training_history.log"
        self.assertFalse(log_file.exists())
        
        self.trainer.log_results(acc=0.95, loss=0.1, elapsed_time=100)
        
        self.assertTrue(log_file.exists())

    def test_cleanup(self):
        """Test cleanup method."""
        self.model.build()
        self.assertIsNotNone(self.model.model)
        
        self.trainer.cleanup()
        
        self.assertIsNone(self.model.model)


if __name__ == '__main__':
    unittest.main()
