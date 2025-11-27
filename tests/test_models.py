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


@pytest.mark.unit
class TestBaseModel:
    """Tests for BaseModel abstract class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = BaseConfig(working_dir=self.temp_dir)

        yield

    def test_base_model_has_required_methods(self):
        """Test that BaseModel has required abstract methods."""
        assert hasattr(BaseModel, "build")
        assert hasattr(BaseModel, "compile")
        assert hasattr(BaseModel, "save")
        assert hasattr(BaseModel, "load")


@pytest.mark.unit
class TestArchitectureFactory:
    """Tests for ArchitectureFactory."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
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


@pytest.mark.unit
class TestArchitectureFactoryEdgeCases:
    """Edge case tests for ArchitectureFactory."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        yield

    def test_create_with_binary_classification(self):
        """Test creating model for binary classification."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=2,
            class_names=["class1", "class2"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        assert model is not None
        assert model.output_shape[-1] == 2

    def test_create_with_many_classes(self):
        """Test creating model with many classes."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=100,
            class_names=[f"class{i}" for i in range(100)],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        assert model is not None
        assert model.output_shape[-1] == 100

    def test_create_with_very_small_image_size(self):
        """Test creating model with very small images."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(32, 32),
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        assert model is not None
        assert model.input_shape[1:] == (32, 32, 3)

    def test_create_with_large_image_size(self):
        """Test creating model with large images."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(256, 256),
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        assert model is not None
        assert model.input_shape[1:] == (256, 256, 3)

    def test_create_with_grayscale_images(self):
        """Test creating model with grayscale images."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            color_channels=1,
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        assert model is not None
        assert model.input_shape[1:] == (64, 64, 1)

    def test_create_with_rectangular_images(self):
        """Test creating model with non-square images."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(128, 64),
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        assert model is not None
        assert model.input_shape[1:] == (128, 64, 3)

    def test_model_handles_batch_prediction(self):
        """Test that model can handle batch predictions."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        # Test with various batch sizes
        for batch_size in [1, 8, 32, 64]:
            dummy_input = np.random.rand(batch_size, *config.input_shape).astype("float32")
            predictions = model.predict(dummy_input, verbose=0)
            assert predictions.shape == (batch_size, config.num_classes)

    def test_model_trainable_parameters(self):
        """Test that model has trainable parameters."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        # Should have trainable parameters
        trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        assert trainable_count > 0

    def test_model_with_extreme_dropout(self):
        """Test model creation with extreme dropout rate."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            dropout_rate=0.9,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        assert model is not None
        # Check that dropout layers exist
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "Dropout" in layer_types

    def test_model_with_no_dropout(self):
        """Test model creation with zero dropout."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            dropout_rate=0.0,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        assert model is not None

    def test_auto_complexity_with_small_dataset(self):
        """Test auto complexity selection prefers simple for small datasets."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=2,
            class_names=["c1", "c2"],
            architecture_complexity="auto",
        )

        model = ArchitectureFactory.create(config, complexity="auto")

        assert model is not None
        # Auto should choose appropriately
        assert model.output_shape[-1] == 2

    def test_auto_complexity_with_many_classes(self):
        """Test auto complexity selection with many classes."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=50,
            class_names=[f"c{i}" for i in range(50)],
            architecture_complexity="auto",
        )

        model = ArchitectureFactory.create(config, complexity="auto")

        assert model is not None
        assert model.output_shape[-1] == 50

    def test_model_output_activation_is_softmax(self):
        """Test that final layer uses softmax activation."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        # Check last layer activation
        last_layer = model.layers[-1]
        assert hasattr(last_layer, "activation")
        # Softmax activation function
        assert last_layer.activation.__name__ == "softmax"

    def test_model_summary_works(self):
        """Test that model.summary() works without errors."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        # Should not raise any errors
        try:
            import io
            import sys

            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            model.summary()
            output = buffer.getvalue()
            sys.stdout = old_stdout

            assert len(output) > 0
            assert "Total params" in output
        except Exception as e:
            pytest.fail(f"model.summary() raised exception: {e}")

    def test_model_is_serializable(self):
        """Test that model can be converted to JSON."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity="simple")

        # Should be able to serialize
        json_config = model.to_json()
        assert len(json_config) > 0
        assert "config" in json_config

    def test_complexity_consistency(self):
        """Test that same complexity produces consistent models."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )

        model1 = ArchitectureFactory.create(config, complexity="medium")
        model2 = ArchitectureFactory.create(config, complexity="medium")

        # Should have same architecture
        assert len(model1.layers) == len(model2.layers)
        assert model1.count_params() == model2.count_params()

        # Check layer types match
        types1 = [type(layer).__name__ for layer in model1.layers]
        types2 = [type(layer).__name__ for layer in model2.layers]
        assert types1 == types2

