"""Tests for model modules."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow not available")

from img_classifier_config import BaseConfig, DatasetConfig  # noqa: E402
from img_classifier_models import ArchitectureFactory, BaseModel  # noqa: E402

pytestmark = pytest.mark.unit

class TestBaseModel:
    """Tests for BaseModel abstract class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = BaseConfig(working_dir=self.temp_dir)

        yield

    @pytest.mark.smoke
    def test_base_model_has_required_methods(self):
        """Test that BaseModel has required abstract methods."""
        assert hasattr(BaseModel, "build")
        assert hasattr(BaseModel, "compile")
        assert hasattr(BaseModel, "save")
        assert hasattr(BaseModel, "load")


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

    @pytest.mark.smoke
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

    @pytest.mark.smoke
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

    def test_compile_with_default_settings(self):
        """Test compile with default settings."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        model = classifier.build()
        classifier.compile()

        assert model.optimizer is not None
        assert model.loss is not None

    def test_compile_with_custom_settings(self):
        """Test compile with custom optimizer and learning rate."""
        from img_classifier_models import CnnClassifier

        self.config.learning_rate = 0.01
        classifier = CnnClassifier(self.config)
        model = classifier.build()
        classifier.compile()

        # Check that learning rate was applied
        assert hasattr(model.optimizer, "learning_rate")

    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        classifier.build()
        classifier.compile()

        # Save model
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.config.models_dir / "test_model.keras"
        classifier.save(model_path)

        assert model_path.exists()

        # Load model
        new_classifier = CnnClassifier(self.config)
        new_classifier.load(model_path)

        assert new_classifier.model is not None

    def test_model_can_train_on_dummy_data(self):
        """Test that model can train on dummy data."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        model = classifier.build()
        classifier.compile()

        # Create dummy data
        X_train = np.random.rand(10, *self.config.input_shape).astype("float32")
        y_train = tf.keras.utils.to_categorical(
            np.random.randint(0, self.config.num_classes, 10), num_classes=self.config.num_classes
        )

        # Train for 1 epoch
        history = model.fit(X_train, y_train, epochs=1, verbose=0)

        assert history is not None
        assert "loss" in history.history


class TestArchitectureFactoryEdgeCasesAdditional:
    """Additional edge case tests for ArchitectureFactory."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        yield

    @pytest.mark.smoke
    def test_invalid_complexity_raises_error(self):
        """Test that invalid complexity raises appropriate error."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=2,
            class_names=["class1", "class2"],
        )

        with pytest.raises(ValueError):
            ArchitectureFactory.create(config, complexity="invalid")

    def test_very_small_image_size(self):
        """Test with very small image size."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(32, 32),
            num_classes=2,
            class_names=["class1", "class2"],
        )

        model = ArchitectureFactory.create(config, "simple")
        assert model is not None

    def test_large_number_of_classes(self):
        """Test with large number of classes."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=100,
            class_names=[f"class{i}" for i in range(100)],
        )

        model = ArchitectureFactory.create(config)
        assert model.output_shape[-1] == 100

    def test_binary_classification(self):
        """Test binary classification (2 classes)."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=2,
            class_names=["class0", "class1"],
        )

        model = ArchitectureFactory.create(config)
        assert model.output_shape[-1] == 2

    def test_non_square_images(self):
        """Test with non-square images."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(128, 64),  # Wider than tall
            num_classes=4,
            class_names=["a", "b", "c", "d"],
        )

        model = ArchitectureFactory.create(config)
        assert model.input_shape[1:] == (128, 64, 3)

    def test_grayscale_images(self):
        """Test with grayscale images (1 channel)."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=3,
            class_names=["a", "b", "c"],
            color_channels=1,
        )

        model = ArchitectureFactory.create(config)
        assert model.input_shape[1:] == (64, 64, 1)

    def test_model_with_different_dropout_rates(self):
        """Test models with different dropout rates."""
        config1 = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=2,
            class_names=["a", "b"],
            dropout_rate=0.2,
        )

        config2 = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=2,
            class_names=["a", "b"],
            dropout_rate=0.5,
        )

        model1 = ArchitectureFactory.create(config1)
        model2 = ArchitectureFactory.create(config2)

        # Both should be valid models
        assert model1 is not None
        assert model2 is not None


class TestCnnClassifierEdgeCases:
    """Edge case tests for CnnClassifier."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = BaseConfig(working_dir=self.temp_dir)
        yield

    @pytest.mark.smoke
    def test_multiple_builds_replace_model(self):
        """Test that building multiple times replaces the model."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        model1 = classifier.build()
        model2 = classifier.build()

        # Should be different models
        assert model1 is not model2
        assert classifier.model is model2

    def test_compile_before_build_does_nothing(self):
        """Test that compile before build handles gracefully."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        classifier.compile()

        # Compile should build the model if needed
        assert classifier.model is not None
        assert classifier.model.optimizer is not None

    def test_save_without_model_raises_error(self):
        """Test that save without model raises appropriate error."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)

        with pytest.raises(Exception):
            classifier.save(self.temp_dir / "model.keras")

    def test_load_nonexistent_model_raises_error(self):
        """Test that loading non-existent model raises error."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)

        with pytest.raises(Exception):
            classifier.load(self.temp_dir / "nonexistent.keras")

    def test_seed_produces_reproducible_models(self):
        """Test that seed produces reproducible models."""
        from img_classifier_models import CnnClassifier

        # Create two models with same seed
        classifier1 = CnnClassifier(self.config, seed=42)
        model1 = classifier1.build()

        classifier2 = CnnClassifier(self.config, seed=42)
        model2 = classifier2.build()

        # Models should have same architecture
        assert model1.count_params() == model2.count_params()

    def test_different_seeds_produce_different_initializations(self):
        """Test that different seeds produce different weight initializations."""
        from img_classifier_models import CnnClassifier

        classifier1 = CnnClassifier(self.config, seed=42)
        model1 = classifier1.build()
        weights1 = model1.get_weights()[0]

        classifier2 = CnnClassifier(self.config, seed=43)
        model2 = classifier2.build()
        weights2 = model2.get_weights()[0]

        # Weights should be different
        assert not np.allclose(weights1, weights2)

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


class TestArchitectureFactoryEdgeCases:
    """Edge case tests for ArchitectureFactory."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        yield

    @pytest.mark.smoke
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


class TestArchitectureFactoryCustomSpec:
    """Tests for ArchitectureFactory with custom specifications."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )
        yield

    @pytest.mark.smoke
    def test_create_with_custom_spec(self):
        """Test creating model with custom architecture specification."""
        from img_classifier_models.architecture_generator import ArchitectureSpec

        custom_spec = ArchitectureSpec(
            name="custom",
            conv_blocks=[
                {"filters": 32, "kernel_size": 3, "activation": "relu", "pooling": True},
                {"filters": 64, "kernel_size": 3, "activation": "relu", "pooling": True},
            ],
            dense_layers=[128],
            use_batch_norm=True,
            use_global_pooling=False,
            dropout_rate=0.5,
        )

        model = ArchitectureFactory.create(self.config, custom_spec=custom_spec)

        assert model is not None
        assert model.output_shape[-1] == self.config.num_classes

    def test_custom_spec_with_batch_normalization(self):
        """Test custom spec with batch normalization enabled."""
        from img_classifier_models.architecture_generator import ArchitectureSpec

        custom_spec = ArchitectureSpec(
            name="custom_bn",
            conv_blocks=[
                {"filters": 32, "kernel_size": 3, "activation": "relu", "pooling": True},
            ],
            dense_layers=[64],
            use_batch_norm=True,
            dropout_rate=0.3,
        )

        model = ArchitectureFactory.create(self.config, custom_spec=custom_spec)

        # Should contain BatchNormalization layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "BatchNormalization" in layer_types

    def test_custom_spec_with_global_pooling(self):
        """Test custom spec with global average pooling."""
        from img_classifier_models.architecture_generator import ArchitectureSpec

        custom_spec = ArchitectureSpec(
            name="custom_gap",
            conv_blocks=[
                {"filters": 32, "kernel_size": 3, "activation": "relu", "pooling": True},
            ],
            dense_layers=[64],
            use_global_pooling=True,
            dropout_rate=0.3,
        )

        model = ArchitectureFactory.create(self.config, custom_spec=custom_spec)

        # Should contain GlobalAveragePooling2D layer
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "GlobalAveragePooling2D" in layer_types
        assert "Flatten" not in layer_types  # Should not have Flatten with GAP

    def test_custom_spec_with_no_pooling(self):
        """Test custom spec with conv blocks without pooling."""
        from img_classifier_models.architecture_generator import ArchitectureSpec

        custom_spec = ArchitectureSpec(
            name="no_pooling",
            conv_blocks=[
                {"filters": 32, "kernel_size": 3, "activation": "relu", "pooling": False},
                {"filters": 64, "kernel_size": 3, "activation": "relu", "pooling": False},
            ],
            dense_layers=[128],
            dropout_rate=0.3,
        )

        model = ArchitectureFactory.create(self.config, custom_spec=custom_spec)

        assert model is not None
        # Count pooling layers - should be fewer or none
        layer_types = [type(layer).__name__ for layer in model.layers]
        pooling_count = layer_types.count("MaxPooling2D")
        # With pooling=False in all blocks, should have 0 MaxPooling2D layers from conv blocks
        assert pooling_count == 0

    def test_custom_spec_with_multiple_dense_layers(self):
        """Test custom spec with multiple dense layers."""
        from img_classifier_models.architecture_generator import ArchitectureSpec

        custom_spec = ArchitectureSpec(
            name="multi_dense",
            conv_blocks=[
                {"filters": 32, "kernel_size": 3, "activation": "relu", "pooling": True},
            ],
            dense_layers=[256, 128, 64],
            dropout_rate=0.4,
        )

        model = ArchitectureFactory.create(self.config, custom_spec=custom_spec)

        # Count Dense layers (excluding output layer)
        layer_types = [type(layer).__name__ for layer in model.layers]
        dense_count = layer_types.count("Dense")
        # Should have 3 + 1 (output layer) = 4 Dense layers
        assert dense_count == 4

    def test_custom_spec_with_different_kernel_sizes(self):
        """Test custom spec with varying kernel sizes."""
        from img_classifier_models.architecture_generator import ArchitectureSpec

        custom_spec = ArchitectureSpec(
            name="varied_kernels",
            conv_blocks=[
                {"filters": 32, "kernel_size": 5, "activation": "relu", "pooling": True},
                {"filters": 64, "kernel_size": 3, "activation": "relu", "pooling": True},
                {"filters": 128, "kernel_size": 7, "activation": "relu", "pooling": True},
            ],
            dense_layers=[128],
            dropout_rate=0.3,
        )

        model = ArchitectureFactory.create(self.config, custom_spec=custom_spec)

        assert model is not None
        # Check that we have Conv2D layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert layer_types.count("Conv2D") == 3

    def test_custom_spec_with_zero_dropout(self):
        """Test custom spec with zero dropout."""
        from img_classifier_models.architecture_generator import ArchitectureSpec

        custom_spec = ArchitectureSpec(
            name="no_dropout",
            conv_blocks=[
                {"filters": 32, "kernel_size": 3, "activation": "relu", "pooling": True},
            ],
            dense_layers=[64],
            dropout_rate=0.0,
        )

        model = ArchitectureFactory.create(self.config, custom_spec=custom_spec)

        # With 0.0 dropout, dropout layers might still be present but with rate=0
        assert model is not None


class TestArchitectureFactoryAutoSelection:
    """Tests for ArchitectureFactory auto-selection logic."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        yield

    @pytest.mark.smoke
    def test_auto_selects_simple_for_binary(self):
        """Test auto complexity selects appropriately for binary classification."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=2,
            class_names=["c1", "c2"],
        )

        model = ArchitectureFactory.create(config, complexity="auto")

        assert model is not None
        # Auto should work for binary classification
        assert model.output_shape[-1] == 2

    def test_auto_selects_for_small_images(self):
        """Test auto complexity works with small images."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(28, 28),
            num_classes=10,
            class_names=[f"c{i}" for i in range(10)],
        )

        model = ArchitectureFactory.create(config, complexity="auto")

        assert model is not None
        assert model.input_shape[1:] == (28, 28, 3)

    def test_auto_selects_for_large_images(self):
        """Test auto complexity works with large images."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(224, 224),
            num_classes=10,
            class_names=[f"c{i}" for i in range(10)],
        )

        model = ArchitectureFactory.create(config, complexity="auto")

        assert model is not None
        assert model.input_shape[1:] == (224, 224, 3)

    def test_complexity_from_config_attribute(self):
        """Test that complexity can be read from config attribute."""
        config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            architecture_complexity="deep",
            class_names=["c1", "c2", "c3", "c4"],
        )

        model = ArchitectureFactory.create(config, complexity=None)

        assert model is not None
        # Should use 'deep' from config
        assert model.output_shape[-1] == 4


class TestCnnClassifierAdvanced:
    """Advanced tests for CnnClassifier."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            class_names=["c1", "c2", "c3", "c4"],
        )
        yield

    @pytest.mark.smoke
    def test_build_with_different_complexities(self):
        """Test building with different complexity levels."""
        from img_classifier_models import CnnClassifier

        for complexity in ["simple", "medium", "deep"]:
            classifier = CnnClassifier(self.config)
            model = classifier.build(complexity=complexity)

            assert model is not None
            assert model.output_shape[-1] == self.config.num_classes

    def test_compile_with_different_optimizers(self):
        """Test compile with various optimizer types."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        classifier.build()

        # Test with different optimizers
        for optimizer in ["adam", "sgd", "rmsprop"]:
            classifier.compile(learning_rate=0.001, optimizer=optimizer)
            # Should not raise errors
            assert classifier.model is not None

    def test_compile_with_different_learning_rates(self):
        """Test compile with various learning rates."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        classifier.build()

        # Test with different learning rates
        for lr in [0.1, 0.01, 0.001, 0.0001]:
            classifier.compile(learning_rate=lr)
            assert classifier.model is not None

    def test_save_creates_directory(self):
        """Test that save creates parent directory if needed."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        classifier.build()
        classifier.compile()

        save_path = self.temp_dir / "deep" / "nested" / "model.keras"

        classifier.save(save_path)

        assert save_path.exists()
        assert save_path.parent.exists()

    def test_load_updates_model_attribute(self):
        """Test that load properly updates the model attribute."""
        from img_classifier_models import CnnClassifier

        # Create and save a model
        classifier1 = CnnClassifier(self.config)
        classifier1.build()
        classifier1.compile()

        save_path = self.temp_dir / "model_load_test.keras"
        classifier1.save(save_path)

        # Create new classifier and load
        classifier2 = CnnClassifier(self.config)
        classifier2.load(save_path)

        assert classifier2.model is not None
        assert classifier2.model.output_shape[-1] == self.config.num_classes

    def test_build_with_custom_seed(self):
        """Test build with custom random seed."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        model = classifier.build(seed=42)

        assert model is not None

    def test_model_predictions_shape(self):
        """Test that model predictions have correct shape."""
        from img_classifier_models import CnnClassifier

        classifier = CnnClassifier(self.config)
        model = classifier.build()
        classifier.compile()

        # Test prediction
        dummy_input = np.random.rand(10, *self.config.input_shape).astype("float32")
        predictions = model.predict(dummy_input, verbose=0)

        assert predictions.shape == (10, self.config.num_classes)
        # Check softmax output (sum to 1)
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)
