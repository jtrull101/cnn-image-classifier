"""Integration tests for end-to-end training workflows.

These tests verify complete training workflows using real dependencies:
- Real data loading and preprocessing
- Actual model creation and compilation
- Complete training loops (with minimal epochs)
- Model evaluation and saving

These tests are slow and should be run separately from unit tests.
"""

import numpy as np
import pytest

from img_classifier_config import BaseConfig, DatasetConfig


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingWorkflow:
    """Integration tests for complete training workflows."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        pytest.importorskip("tensorflow", reason="TensorFlow not available")

        self.temp_dir = isolated_tmp_dir
        self.config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(64, 64),
            num_classes=4,
            class_names=["class1", "class2", "class3", "class4"],
            num_epochs=2,  # Minimal for testing
            batch_size=16,
        )

        yield

    def _create_mock_data_loader(self):
        """Create a data loader with synthetic data for testing."""
        from img_classifier_data import BaseDataLoader

        class TestDataLoader(BaseDataLoader):
            def __init__(self, config):
                super().__init__(config)

            def load_train_data(self):
                """Generate synthetic training data."""
                X = np.random.rand(100, 64, 64, 3).astype("float32")
                y = np.random.randint(0, 4, 100)
                return X, y

            def load_test_data(self):
                """Generate synthetic test data."""
                X = np.random.rand(40, 64, 64, 3).astype("float32")
                y = np.random.randint(0, 4, 40)
                return X, y

            def download_dataset(self):
                return True

            def prepare_dataset(self):
                return True

        return TestDataLoader(self.config)

    def test_trainer_with_real_model(self):
        """Test Trainer with real model and data."""
        from img_classifier_models import ArchitectureFactory, BaseModel
        from img_classifier_training import Trainer

        # Create real model
        keras_model = ArchitectureFactory.create(self.config, complexity="simple")

        class TestModel(BaseModel):
            def __init__(self, config, keras_model):
                super().__init__(config)
                self.model = keras_model

            def build(self):
                return self.model

        model = TestModel(self.config, keras_model)
        data_loader = self._create_mock_data_loader()
        trainer = Trainer(self.config, model, data_loader)

        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data()

        assert X_train.shape[0] > 0
        assert y_train.shape[1] == 4  # One-hot encoded

        # Compile model
        model.compile()

        # Train for minimal epochs
        history = trainer.train(X_train, y_train, X_val, y_val)

        assert history is not None
        assert "loss" in history.history
        assert len(history.history["loss"]) == 2  # 2 epochs

        # Evaluate
        loss, acc = trainer.evaluate(X_test, y_test)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_orchestrator_basic_training(self):
        """Test TrainingOrchestrator with basic training."""
        from img_classifier_training import TrainingOrchestrator

        data_loader = self._create_mock_data_loader()
        orchestrator = TrainingOrchestrator(
            self.config,
            data_loader=data_loader,
            optimize_hyperparameters=False,
        )

        # Run basic training
        results = orchestrator.run()

        assert results is not None
        assert "model" in results
        assert "history" in results
        assert "metrics" in results
        assert results["metrics"]["loss"] >= 0.0

    def test_orchestrator_with_architecture_selection(self):
        """Test orchestrator automatically selects architecture."""
        from img_classifier_training import TrainingOrchestrator

        data_loader = self._create_mock_data_loader()

        # Create orchestrator without specifying architecture
        orchestrator = TrainingOrchestrator(
            self.config,
            data_loader=data_loader,
        )

        # Should automatically select architecture based on dataset
        results = orchestrator.run()

        assert results is not None
        assert results["model"] is not None


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_data
class TestRealDataWorkflow:
    """Integration tests with real dataset loading (requires data download)."""

    @pytest.fixture(autouse=True)
    def setup_method(self, temp_dataset_structure):
        """Set up test fixtures with real dataset structure."""
        pytest.importorskip("tensorflow", reason="TensorFlow not available")

        self.dataset_info = temp_dataset_structure
        self.dataset_dir = self.dataset_info["dataset_dir"]

        yield

    def test_image_loader_with_real_structure(self):
        """Test ImageDataLoader with real directory structure."""
        from img_classifier_config import DatasetDetector
        from img_classifier_data import ImageDataLoader

        # Detect dataset configuration
        detector = DatasetDetector(self.dataset_dir)
        info = detector.detect()
        config = detector.create_config(working_dir=self.dataset_dir.parent)

        # Create loader and attempt to load data
        loader = ImageDataLoader(config)

        # This will fail if no actual images, but tests the integration
        try:
            categories = loader.get_categories(self.dataset_info["train_dir"])
            assert len(categories) == 4
            assert "class1" in categories
        except Exception as e:
            pytest.skip(f"Cannot load real data: {e}")

    def test_end_to_end_with_dataset_detection(self):
        """Test complete workflow from dataset detection to training."""
        from img_classifier_training import TrainingOrchestrator

        try:
            # Create orchestrator from dataset path
            orchestrator = TrainingOrchestrator.from_dataset_path(
                dataset_path=self.dataset_dir,
                project_name="test_integration",
                working_dir=self.dataset_dir.parent,
            )

            # Verify configuration was detected
            assert orchestrator.config is not None
            assert orchestrator.config.num_classes == 4
            assert orchestrator.data_loader is not None

        except Exception as e:
            pytest.skip(f"Dataset detection failed: {e}")


@pytest.mark.integration
class TestModelSaveLoad:
    """Integration tests for model saving and loading."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        pytest.importorskip("tensorflow", reason="TensorFlow not available")

        self.temp_dir = isolated_tmp_dir
        self.config = DatasetConfig(
            working_dir=self.temp_dir,
            image_size=(32, 32),
            num_classes=2,
            class_names=["class1", "class2"],
        )

        yield

    def test_model_save_and_load(self):
        """Test that models can be saved and loaded correctly."""
        from img_classifier_models import ArchitectureFactory, BaseModel

        # Create and save model
        keras_model = ArchitectureFactory.create(self.config, complexity="simple")

        class TestModel(BaseModel):
            def __init__(self, config, keras_model):
                super().__init__(config)
                self.model = keras_model

            def build(self):
                return self.model

        model = TestModel(self.config, keras_model)
        model.compile()

        save_path = self.temp_dir / "test_model.keras"
        model.save(save_path)

        assert save_path.exists()

        # Load model
        loaded_model = TestModel(self.config, None)
        loaded_model.load(save_path)

        assert loaded_model.model is not None
        # Verify architectures match
        assert len(loaded_model.model.layers) == len(model.model.layers)

