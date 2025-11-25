"""Tests for model modules."""

from pathlib import Path

import numpy as np
import pytest


try:
    import tensorflow as tf

    from img_classifier_models import ArchitectureFactory, BaseModel
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    BaseModel = None
    ArchitectureFactory = None

from img_classifier_config import BaseConfig, DatasetConfig


pytestmark = pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")


class TestBaseModel:
    """Tests for BaseModel abstract class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures."""
        self.temp_dir = tmp_path
        self.config = BaseConfig(working_dir=self.temp_dir)

        yield


    def test_base_model_has_required_methods(self):
        """Test that BaseModel has required abstract methods."""
        assert hasattr(BaseModel, 'build')
        assert hasattr(BaseModel, 'compile')
        assert hasattr(BaseModel, 'save')
        assert hasattr(BaseModel, 'load')


class TestArchitectureFactory:
    """Tests for ArchitectureFactory."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures."""
        self.temp_dir = tmp_path
        self.config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            class_names=["class1", "class2", "class3", "class4"]
        )

        yield


    def test_create_simple_architecture(self):
        """Test creating simple architecture."""
        model = ArchitectureFactory.create(self.config, complexity="simple")

        assert model is not None
        assert model.input_shape[1:] == self.config.input_shape
        assert model.output_shape[-1] == self.config.num_classes

    def test_create_medium_architecture(self):
        """Test creating medium architecture."""
        model = ArchitectureFactory.create(self.config, complexity="medium")

        assert model is not None
        assert model.input_shape[1:] == self.config.input_shape
        assert model.output_shape[-1] == self.config.num_classes

    def test_create_deep_architecture(self):
        """Test creating deep architecture."""
        model = ArchitectureFactory.create(self.config, complexity="deep")

        assert model is not None
        assert model.input_shape[1:] == self.config.input_shape
        assert model.output_shape[-1] == self.config.num_classes

    def test_auto_complexity_selection(self):
        """Test automatic complexity selection."""
        model = ArchitectureFactory.create(self.config, complexity="auto")

        assert model is not None
        assert model.output_shape[-1] == self.config.num_classes

    def test_model_has_expected_layers(self):
        """Test that model has expected layer structure."""
        model = ArchitectureFactory.create(self.config, complexity="simple")

        # Should have Conv2D, MaxPooling2D, Flatten, and Dense layers
        layer_types = [type(layer).__name__ for layer in model.layers]

        assert 'Conv2D' in layer_types
        assert 'MaxPooling2D' in layer_types
        assert 'Flatten' in layer_types
        assert 'Dense' in layer_types

    def test_model_accepts_correct_input_shape(self):
        """Test that model accepts input with correct shape."""
        model = ArchitectureFactory.create(self.config)
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Create dummy input
        batch_size = 2
        dummy_input = np.random.rand(batch_size, *self.config.input_shape).astype('float32')

        # Model should be able to predict on this input
        predictions = model.predict(dummy_input, verbose=0)

        assert predictions.shape == (batch_size, self.config.num_classes)

    def test_output_is_probability_distribution(self):
        """Test that model output is a probability distribution."""
        model = ArchitectureFactory.create(self.config)
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        dummy_input = np.random.rand(1, *self.config.input_shape).astype('float32')
        predictions = model.predict(dummy_input, verbose=0)

        # Output should sum to approximately 1 (softmax)
        assert predictions.sum() == pytest.approx(1.0, abs=1e-5)

        # All values should be between 0 and 1
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)

    def test_different_complexities_different_sizes(self):
        """Test that different complexities produce different model sizes."""
        simple_model = ArchitectureFactory.create(self.config, complexity="simple")
        deep_model = ArchitectureFactory.create(self.config, complexity="deep")

        # Deep model should have more layers
        assert len(deep_model.layers) > len(simple_model.layers)
