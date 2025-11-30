"""Unit tests for Training Orchestrator using mocking.

These tests verify TrainingOrchestrator behavior with mocked dependencies
to ensure fast execution and isolation from other packages.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# TODO remove this dependency - should be just /training/ items
from img_classifier_config import DatasetConfig, BaseConfig
from img_classifier_training import TrainingOrchestrator

pytestmark = pytest.mark.unit


class TestTrainingOrchestrator:
    """Unit tests for TrainingOrchestrator class with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures with mocked dependencies."""
        self.temp_dir = isolated_tmp_dir

        # Create mock config
        self.mock_config = Mock()
        self.mock_config.working_dir = self.temp_dir
        self.mock_config.num_classes = 4
        self.mock_config.input_shape = (128, 128, 3)
        self.mock_config.models_dir = self.temp_dir / "models"

        # Real config for scenarios that require concrete behavior
        self.config = BaseConfig(working_dir=self.temp_dir)

        # Mock data loader
        self.mock_loader = Mock()
        self.mock_loader.config = self.config

        yield

    def test_initialization_default(self):
        """Test orchestrator initialization with defaults."""
        with patch("img_classifier_training.orchestrator.ImageDataLoader"):
            from img_classifier_training import TrainingOrchestrator

            orchestrator = TrainingOrchestrator(self.mock_config)

            assert orchestrator.config == self.mock_config
            assert orchestrator.data_loader is not None
            assert orchestrator.optimize_hyperparameters is False
            assert orchestrator.optimizer_type == "random"
            assert orchestrator.search_space is None
            assert orchestrator.max_trials == 20
            assert orchestrator.optimizer is None
            assert orchestrator.best_model is None
            assert orchestrator.best_config is None

    def test_initialization_with_data_loader(self):
        """Test orchestrator initialization with custom data loader."""
        from img_classifier_training import TrainingOrchestrator

        orchestrator = TrainingOrchestrator(self.mock_config, data_loader=self.mock_loader)

        assert orchestrator.data_loader == self.mock_loader

    def test_initialization_with_optimization(self):
        """Test orchestrator initialization with hyperparameter optimization."""
        from img_classifier_training import TrainingOrchestrator

        orchestrator = TrainingOrchestrator(
            self.mock_config,
            data_loader=self.mock_loader,
            optimize_hyperparameters=True,
            optimizer_type="bayesian",
            max_trials=50,
        )

        assert orchestrator.optimize_hyperparameters is True
        assert orchestrator.optimizer_type == "bayesian"
        assert orchestrator.max_trials == 50

    @patch("img_classifier_training.orchestrator.DatasetDetector")
    def test_from_dataset_path(self, mock_detector_class):
        """Test creating orchestrator from dataset path."""
        # Setup mock
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector

        mock_detector.create_config.return_value = self.config
        mock_detector.detect.return_value = {
            "dataset_path": self.temp_dir,
            "num_classes": 4,
            "class_names": ["class1", "class2", "class3", "class4"],
            "total_images": 100,
            "is_balanced": True,
            "recommended_complexity": "medium",
        }

        # Create orchestrator
        orchestrator = TrainingOrchestrator.from_dataset_path(
            dataset_path=self.temp_dir, project_name="test_project", working_dir=self.temp_dir
        )

        # Verify
        mock_detector_class.assert_called_once_with(self.temp_dir)
        mock_detector.create_config.assert_called_once()
        mock_detector.detect.assert_called_once()
        assert orchestrator.config == self.config

    @patch("img_classifier_training.orchestrator.DatasetDetector")
    def test_from_dataset_path_with_kwargs(self, mock_detector_class):
        """Test from_dataset_path with additional kwargs."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.create_config.return_value = self.config
        mock_detector.detect.return_value = {
            "dataset_path": self.temp_dir,
            "num_classes": 2,
            "class_names": ["cat", "dog"],
            "total_images": 50,
            "is_balanced": False,
            "recommended_complexity": "simple",
        }

        orchestrator = TrainingOrchestrator.from_dataset_path(
            dataset_path=self.temp_dir, optimize_hyperparameters=True, optimizer_type="grid"
        )

        assert orchestrator.optimize_hyperparameters is True
        assert orchestrator.optimizer_type == "grid"

    def test_prepare_dataset_with_download(self):
        """Test _prepare_dataset when download is needed."""
        self.mock_loader.download_dataset = Mock()
        self.mock_loader.prepare_dataset = Mock(return_value=True)

        orchestrator = TrainingOrchestrator(self.config, data_loader=self.mock_loader)

        result = orchestrator._prepare_dataset()

        assert result is True
        self.mock_loader.download_dataset.assert_called_once()
        self.mock_loader.prepare_dataset.assert_called_once()

    def test_prepare_dataset_without_download(self):
        """Test _prepare_dataset when download method doesn't exist."""
        # Mock loader without download_dataset method
        mock_loader_no_download = Mock(spec=["prepare_dataset"])
        mock_loader_no_download.prepare_dataset = Mock(return_value=True)

        orchestrator = TrainingOrchestrator(self.config, data_loader=mock_loader_no_download)

        result = orchestrator._prepare_dataset()

        assert result is True
        mock_loader_no_download.prepare_dataset.assert_called_once()

    def test_prepare_dataset_failure(self):
        """Test _prepare_dataset when preparation fails."""
        self.mock_loader.prepare_dataset = Mock(return_value=False)

        orchestrator = TrainingOrchestrator(self.config, data_loader=self.mock_loader)

        result = orchestrator._prepare_dataset()

        assert result is False

    def test_prepare_dataset_no_methods(self):
        """Test _prepare_dataset when loader has no prepare methods."""
        mock_loader_minimal = Mock(spec=["config"])
        mock_loader_minimal.config = self.config

        orchestrator = TrainingOrchestrator(self.config, data_loader=mock_loader_minimal)

        result = orchestrator._prepare_dataset()

        assert result is True

    @patch("img_classifier_training.orchestrator.ArchitectureFactory")
    @patch("img_classifier_training.orchestrator.Trainer")
    def test_run_single_training(self, mock_trainer_class, mock_arch_factory):
        """Test _run_single_training method."""
        # Setup mocks
        mock_model = Mock()
        mock_arch_factory.create.return_value = mock_model

        mock_trainer = Mock()
        mock_trainer.run.return_value = (0.95, 0.15)
        mock_trainer_class.return_value = mock_trainer

        orchestrator = TrainingOrchestrator(self.config, data_loader=self.mock_loader)

        # Run training
        orchestrator._run_single_training(plot=False)

        # Verify
        mock_arch_factory.create.assert_called_once_with(self.config)
        mock_trainer_class.assert_called_once()
        mock_trainer.run.assert_called_once_with(plot=False, force_save=True)
        assert orchestrator.best_model is not None
        assert orchestrator.best_config == self.config

    @patch("img_classifier_training.orchestrator.ArchitectureFactory")
    @patch("img_classifier_training.orchestrator.Trainer")
    def test_run_single_training_with_plot(self, mock_trainer_class, mock_arch_factory):
        """Test _run_single_training with plotting enabled."""
        mock_model = Mock()
        mock_arch_factory.create.return_value = mock_model

        mock_trainer = Mock()
        mock_trainer.run.return_value = (0.98, 0.08)
        mock_trainer_class.return_value = mock_trainer

        orchestrator = TrainingOrchestrator(self.config, data_loader=self.mock_loader)

        orchestrator._run_single_training(plot=True)

        mock_trainer.run.assert_called_once_with(plot=True, force_save=True)

    @patch("img_classifier_training.orchestrator.create_optimizer")
    def test_run_optimization_initialization(self, mock_create_optimizer):
        """Test _run_optimization initialization."""
        mock_optimizer = Mock()
        mock_optimizer.should_stop.return_value = True  # Stop immediately
        mock_optimizer.best_result = None
        mock_optimizer.get_summary.return_value = "Summary"
        mock_create_optimizer.return_value = mock_optimizer

        orchestrator = TrainingOrchestrator(
            self.config, data_loader=self.mock_loader, optimize_hyperparameters=True, max_trials=10
        )

        orchestrator._run_optimization(plot=False)

        mock_create_optimizer.assert_called_once_with("random", self.config, None, max_trials=10)
        assert orchestrator.optimizer == mock_optimizer

    def test_create_trial_config_base_config(self):
        """Test _create_trial_config with BaseConfig."""
        hyperparams = {
            "learning_rate": 0.002,
            "batch_size": 64,
            "dropout_rate": 0.5,
            "num_epochs": 30,
            "architecture": "deep",
        }

        orchestrator = TrainingOrchestrator(self.config, data_loader=self.mock_loader)

        trial_config = orchestrator._create_trial_config(hyperparams)

        assert isinstance(trial_config, BaseConfig)
        assert trial_config.learning_rate == 0.002
        assert trial_config.batch_size == 64
        assert trial_config.dropout_rate == 0.5
        assert trial_config.num_epochs == 30

    def test_create_trial_config_dataset_config(self):
        """Test _create_trial_config with DatasetConfig."""
        dataset_config = DatasetConfig(
            working_dir=self.temp_dir, num_classes=4, class_names=["a", "b", "c", "d"]
        )

        hyperparams = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout_rate": 0.3,
            "num_epochs": 20,
            "architecture": "medium",
        }

        orchestrator = TrainingOrchestrator(dataset_config, data_loader=self.mock_loader)

        trial_config = orchestrator._create_trial_config(hyperparams)

        assert isinstance(trial_config, DatasetConfig)
        assert trial_config.num_classes == 4
        assert trial_config.learning_rate == 0.001

    @patch("img_classifier_training.orchestrator.ArchitectureFactory")
    @patch("img_classifier_training.orchestrator.Trainer")
    @patch("img_classifier_training.orchestrator.time")
    def test_run_trial_success(self, mock_time, mock_trainer_class, mock_arch_factory):
        """Test successful _run_trial execution."""
        # Mock time
        mock_time.time.side_effect = [1000.0, 1100.0]  # 100s training time

        # Mock model creation
        mock_model = Mock()
        mock_arch_factory.create.return_value = mock_model

        # Mock trainer
        mock_trainer = Mock()
        mock_history = Mock()
        mock_history.history = {
            "acc": [0.7, 0.8, 0.9],
            "loss": [0.5, 0.3, 0.2],
            "val_acc": [0.65, 0.75, 0.85],
            "val_loss": [0.6, 0.4, 0.25],
        }
        mock_trainer.prepare_data.return_value = (
            np.array([1, 2, 3]),
            np.array([0, 1, 0]),  # X_train, y_train
            np.array([4, 5]),
            np.array([1, 0]),  # X_val, y_val
            np.array([6, 7]),
            np.array([0, 1]),  # X_test, y_test
        )
        mock_trainer.train.return_value = mock_history
        mock_trainer.evaluate.return_value = (0.22, 0.88)  # test_loss, test_acc
        mock_trainer.cleanup = Mock()
        mock_trainer_class.return_value = mock_trainer

        orchestrator = TrainingOrchestrator(self.config, data_loader=self.mock_loader)
        orchestrator.optimizer = Mock()
        orchestrator.optimizer.best_result = None

        trial_config = BaseConfig(working_dir=self.temp_dir)
        result = orchestrator._run_trial(trial_config, 1, plot=False)

        # Verify result
        assert result.trial_id == 1
        assert result.train_accuracy == 0.9
        assert result.val_accuracy == 0.85
        assert result.test_accuracy == 0.88
        assert result.training_time == 100.0
        assert result.train_loss == 0.2
        assert result.val_loss == 0.25
        assert result.test_loss == 0.22

        # Verify cleanup was called
        mock_trainer.cleanup.assert_called_once()

    @patch("img_classifier_training.orchestrator.ArchitectureFactory")
    @patch("img_classifier_training.orchestrator.Trainer")
    @patch("img_classifier_training.orchestrator.time")
    def test_run_trial_saves_best_model(self, mock_time, mock_trainer_class, mock_arch_factory):
        """Test that best model is saved during trials."""
        mock_time.time.side_effect = [1000.0, 1050.0]

        mock_model = Mock()
        mock_model.save = Mock()
        mock_arch_factory.create.return_value = mock_model

        mock_trainer = Mock()
        mock_history = Mock()
        mock_history.history = {"acc": [0.95], "loss": [0.1], "val_acc": [0.92], "val_loss": [0.15]}
        mock_trainer.prepare_data.return_value = (
            np.array([1]),
            np.array([0]),
            np.array([2]),
            np.array([1]),
            np.array([3]),
            np.array([0]),
        )
        mock_trainer.train.return_value = mock_history
        mock_trainer.evaluate.return_value = (0.12, 0.93)
        mock_trainer.cleanup = Mock()
        mock_trainer_class.return_value = mock_trainer

        # Create models directory
        models_dir = self.temp_dir / "models"
        models_dir.mkdir()

        trial_config = BaseConfig(working_dir=self.temp_dir)
        orchestrator = TrainingOrchestrator(self.config, data_loader=self.mock_loader)
        orchestrator.optimizer = Mock()
        orchestrator.optimizer.best_result = None

        orchestrator._run_trial(trial_config, 1, plot=False)

        # Verify best model was saved
        assert orchestrator.best_model is not None
        assert orchestrator.best_config == trial_config
        mock_model.save.assert_called_once()

    @patch("img_classifier_training.orchestrator.ArchitectureFactory")
    @patch("img_classifier_training.orchestrator.time")
    def test_run_trial_handles_exception(self, mock_time, mock_arch_factory):
        """Test _run_trial handles exceptions gracefully."""
        mock_time.time.return_value = 1000.0
        mock_arch_factory.create.side_effect = Exception("Model creation failed")

        trial_config = BaseConfig(working_dir=self.temp_dir)
        orchestrator = TrainingOrchestrator(self.config, data_loader=self.mock_loader)

        result = orchestrator._run_trial(trial_config, 1, plot=False)

        # Should return a failed result
        assert result.trial_id == 1
        assert result.train_accuracy == 0.0
        assert result.val_accuracy == 0.0
        assert result.test_accuracy == 0.0

    @patch.object(TrainingOrchestrator, "_prepare_dataset")
    @patch.object(TrainingOrchestrator, "_run_single_training")
    def test_run_without_optimization(self, mock_run_single, mock_prepare):
        """Test run method without hyperparameter optimization."""
        mock_prepare.return_value = True
        mock_model = Mock()
        mock_run_single.return_value = {"model": mock_model}

        orchestrator = TrainingOrchestrator(
            self.config, data_loader=self.mock_loader, optimize_hyperparameters=False
        )
        orchestrator.best_model = mock_model

        result = orchestrator.run(plot=False)

        mock_prepare.assert_called_once()
        mock_run_single.assert_called_once_with(plot=False)
        assert result["model"] == mock_model

    @patch.object(TrainingOrchestrator, "_prepare_dataset")
    @patch.object(TrainingOrchestrator, "_run_optimization")
    def test_run_with_optimization(self, mock_run_opt, mock_prepare):
        """Test run method with hyperparameter optimization."""
        mock_prepare.return_value = True
        mock_model = Mock()
        mock_run_opt.return_value = {"model": mock_model}

        orchestrator = TrainingOrchestrator(
            self.config, data_loader=self.mock_loader, optimize_hyperparameters=True
        )
        orchestrator.best_model = mock_model

        result = orchestrator.run(plot=True)

        mock_prepare.assert_called_once()
        mock_run_opt.assert_called_once_with(plot=True)
        assert result["model"] == mock_model

    @patch.object(TrainingOrchestrator, "_prepare_dataset")
    def test_run_preparation_failure(self, mock_prepare):
        """Test run method when dataset preparation fails."""
        mock_prepare.return_value = False

        orchestrator = TrainingOrchestrator(self.config, data_loader=self.mock_loader)

        with pytest.raises(RuntimeError, match="Failed to prepare dataset"):
            orchestrator.run()

    def test_multiple_trials_updates_best_result(self):
        """Test that multiple trials properly track the best result."""
        # This is an integration-style test for the optimization loop
        orchestrator = TrainingOrchestrator(
            self.config, data_loader=self.mock_loader, optimize_hyperparameters=True, max_trials=3
        )

        # Verify initial state
        assert orchestrator.best_model is None
        assert orchestrator.best_config is None


@pytest.mark.unit
class TestTrainingOrchestratorEdgeCases:
    """Edge case tests for TrainingOrchestrator."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = BaseConfig(working_dir=self.temp_dir)
        yield

    def test_invalid_optimizer_type(self):
        """Test with invalid optimizer type."""
        orchestrator = TrainingOrchestrator(
            self.config, optimize_hyperparameters=True, optimizer_type="invalid_type"
        )

        assert orchestrator.optimizer_type == "invalid_type"

    def test_zero_max_trials(self):
        """Test with zero max trials."""
        orchestrator = TrainingOrchestrator(
            self.config, optimize_hyperparameters=True, max_trials=0
        )

        assert orchestrator.max_trials == 0

    def test_negative_max_trials(self):
        """Test with negative max trials."""
        orchestrator = TrainingOrchestrator(
            self.config, optimize_hyperparameters=True, max_trials=-1
        )

        assert orchestrator.max_trials == -1

    def test_none_search_space(self):
        """Test with None search space."""
        orchestrator = TrainingOrchestrator(
            self.config, optimize_hyperparameters=True, search_space=None
        )

        assert orchestrator.search_space is None

    @patch("img_classifier_training.orchestrator.DatasetDetector")
    def test_from_dataset_path_no_project_name(self, mock_detector_class):
        """Test from_dataset_path without project name."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.create_config.return_value = self.config
        mock_detector.detect.return_value = {
            "dataset_path": self.temp_dir,
            "num_classes": 2,
            "class_names": ["a", "b"],
            "total_images": 10,
            "is_balanced": True,
            "recommended_complexity": "simple",
        }

        TrainingOrchestrator.from_dataset_path(dataset_path=self.temp_dir)

        mock_detector.create_config.assert_called_once_with(project_name=None, working_dir=None)

    def test_config_model_dump_compatibility(self):
        """Test that config.model_dump() works for trial config creation."""
        orchestrator = TrainingOrchestrator(self.config)

        hyperparams = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout_rate": 0.3,
            "num_epochs": 25,
            "architecture": "medium",
        }

        # This should not raise an exception
        trial_config = orchestrator._create_trial_config(hyperparams)

        assert trial_config is not None
        assert isinstance(trial_config, BaseConfig)
