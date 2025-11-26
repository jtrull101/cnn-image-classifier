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
        assert hasattr(BaseModel, "build")
        assert hasattr(BaseModel, "compile")
        assert hasattr(BaseModel, "save")
        assert hasattr(BaseModel, "load")


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
            class_names=["class1", "class2", "class3", "class4"],
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

        assert "Conv2D" in layer_types
        assert "MaxPooling2D" in layer_types
        assert "Flatten" in layer_types
        assert "Dense" in layer_types

    def test_model_accepts_correct_input_shape(self):
        """Test that model accepts input with correct shape."""
        model = ArchitectureFactory.create(self.config)
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        # Create dummy input
        batch_size = 2
        dummy_input = np.random.rand(batch_size, *self.config.input_shape).astype("float32")

        # Model should be able to predict on this input
        predictions = model.predict(dummy_input, verbose=0)

        assert predictions.shape == (batch_size, self.config.num_classes)

    def test_output_is_probability_distribution(self):
        """Test that model output is a probability distribution."""
        model = ArchitectureFactory.create(self.config)
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        dummy_input = np.random.rand(1, *self.config.input_shape).astype("float32")
        predictions = model.predict(dummy_input, verbose=0)

        # Output should sum to approximately 1 (softmax)
        assert predictions.sum() == pytest.approx(1.0, abs=1e-5)

        # All values should be between 0 and 1
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)

    def test_different_complexities_different_sizes(self):
        """Test that different complexities produce different model sizes."""
        simple = ArchitectureFactory.create(self.config, "simple")
        medium = ArchitectureFactory.create(self.config, "medium")
        deep = ArchitectureFactory.create(self.config, "deep")

        # Count trainable parameters
        simple_params = simple.count_params()
        medium_params = medium.count_params()
        deep_params = deep.count_params()

        # Should have increasing complexity
        assert simple_params < medium_params < deep_params


class TestCnnClassifier:
    """Tests for CnnClassifier."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures."""
        self.temp_dir = tmp_path
        self.config = BaseConfig(working_dir=self.temp_dir)

        yield

    def test_initialization(self):
        """Test CnnClassifier initialization."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config, seed=42)
        assert classifier.config == self.config
        assert classifier.seed == 42
        assert classifier.model is None

    def test_build_creates_model(self):
        """Test that build() creates a valid model."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        model = classifier.build()

        assert model is not None
        assert isinstance(model, tf.keras.Model)
        assert classifier.model is not None

    def test_model_input_shape(self):
        """Test that model accepts correct input shape."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        model = classifier.build()

        # Check input shape
        expected_shape = (None,) + self.config.input_shape
        assert model.input_shape == expected_shape

    def test_model_output_shape(self):
        """Test that model output matches num_classes."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        model = classifier.build()

        # Check output shape
        assert model.output_shape == (None, self.config.num_classes)

    def test_compile_and_predict(self):
        """Test that compiled model can make predictions."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        model = classifier.build()
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        # Make a prediction with dummy data
        dummy_input = np.random.rand(1, *self.config.input_shape).astype("float32")
        predictions = model.predict(dummy_input, verbose=0)

        assert predictions.shape == (1, self.config.num_classes)
        # Output should be probabilities summing to 1
        assert abs(predictions.sum() - 1.0) < 0.01

    def test_model_has_dropout(self):
        """Test that model includes dropout layers."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        model = classifier.build()

        # Check for dropout layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "Dropout" in layer_types

    def test_model_layer_count(self):
        """Test that model has expected number of layers."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        model = classifier.build()

        # Should have multiple layers (conv, pool, dropout, dense, etc.)
        assert len(model.layers) >= 10

    def test_different_seeds_produce_same_architecture(self):
        """Test that different seeds produce same architecture."""
        from img_classifier_models import CnnClassifier

        classifier1 = CnnClassifier(self.config, seed=42)
        classifier2 = CnnClassifier(self.config, seed=123)

        model1 = classifier1.build()
        model2 = classifier2.build()

        # Same architecture (layer count and shapes)
        assert len(model1.layers) == len(model2.layers)
        assert model1.count_params() == model2.count_params()
