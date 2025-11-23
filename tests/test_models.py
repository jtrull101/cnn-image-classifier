"""Tests for model modules."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np


try:
    import tensorflow as tf

    from alz_mri_models import BaseModel, CNNClassifier
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    BaseModel = None
    CNNClassifier = None

from alz_mri_config import BaseConfig


@unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
class TestBaseModel(unittest.TestCase):
    """Tests for BaseModel abstract class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = BaseConfig(working_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_base_model_has_required_methods(self):
        """Test that BaseModel has required abstract methods."""
        self.assertTrue(hasattr(BaseModel, 'build'))
        self.assertTrue(hasattr(BaseModel, 'compile'))
        self.assertTrue(hasattr(BaseModel, 'save'))
        self.assertTrue(hasattr(BaseModel, 'load'))


@unittest.skipIf(not TENSORFLOW_AVAILABLE, "TensorFlow not available")
class TestCNNClassifier(unittest.TestCase):
    """Tests for CNNClassifier model."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = BaseConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4
        )
        self.model = CNNClassifier(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test CNNClassifier initialization."""
        self.assertIsInstance(self.model, BaseModel)
        self.assertEqual(self.model.config, self.config)
        self.assertEqual(self.model.seed, 1234)
        self.assertIsNone(self.model.model)

    def test_custom_seed(self):
        """Test CNNClassifier with custom seed."""
        model = CNNClassifier(self.config, seed=42)
        self.assertEqual(model.seed, 42)

    def test_build_creates_model(self):
        """Test that build creates a Keras model."""
        keras_model = self.model.build()

        self.assertIsNotNone(keras_model)
        self.assertEqual(keras_model.input_shape[1:], self.config.input_shape)
        self.assertEqual(keras_model.output_shape[-1], self.config.num_classes)

    def test_build_sets_model_attribute(self):
        """Test that build sets the model attribute."""
        self.assertIsNone(self.model.model)
        self.model.build()
        self.assertIsNotNone(self.model.model)

    def test_model_layers(self):
        """Test that model has expected layer structure."""
        keras_model = self.model.build()

        # Should have Conv2D, MaxPooling2D, Dropout, Flatten, and Dense layers
        layer_types = [type(layer).__name__ for layer in keras_model.layers]

        self.assertIn('Conv2D', layer_types)
        self.assertIn('MaxPooling2D', layer_types)
        self.assertIn('Dropout', layer_types)
        self.assertIn('Flatten', layer_types)
        self.assertIn('Dense', layer_types)

    def test_compile_with_defaults(self):
        """Test model compilation with default parameters."""
        self.model.compile()

        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.model.optimizer)
        self.assertEqual(self.model.model.loss, 'categorical_crossentropy')

    def test_compile_creates_model_if_needed(self):
        """Test that compile creates model if not already built."""
        self.assertIsNone(self.model.model)
        self.model.compile()
        self.assertIsNotNone(self.model.model)

    def test_summary_builds_model(self):
        """Test that summary builds model if not already built."""
        self.assertIsNone(self.model.model)
        self.model.summary()
        self.assertIsNotNone(self.model.model)

    def test_save_requires_built_model(self):
        """Test that save raises error if model not built."""
        save_path = self.temp_dir / "model.keras"

        with self.assertRaises(ValueError):
            self.model.save(save_path)

    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        self.model.build()
        self.model.compile()

        save_dir = self.temp_dir / "subdir" / "models"
        save_path = save_dir / "model.keras"

        self.assertFalse(save_dir.exists())
        self.model.save(save_path)
        self.assertTrue(save_path.exists())

    def test_model_accepts_correct_input_shape(self):
        """Test that model accepts input with correct shape."""
        self.model.build()
        self.model.compile()

        # Create dummy input
        batch_size = 2
        dummy_input = np.random.rand(batch_size, *self.config.input_shape)

        # Model should be able to predict on this input
        predictions = self.model.model.predict(dummy_input, verbose=0)

        self.assertEqual(predictions.shape, (batch_size, self.config.num_classes))

    def test_output_is_probability_distribution(self):
        """Test that model output is a probability distribution."""
        self.model.build()
        self.model.compile()

        dummy_input = np.random.rand(1, *self.config.input_shape)
        predictions = self.model.model.predict(dummy_input, verbose=0)

        # Output should sum to approximately 1 (softmax)
        self.assertAlmostEqual(predictions.sum(), 1.0, places=5)

        # All values should be between 0 and 1
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))


if __name__ == '__main__':
    unittest.main()
