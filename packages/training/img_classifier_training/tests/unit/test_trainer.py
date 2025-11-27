"""Unit tests for Trainer module using mocking.

These tests verify Trainer behavior with mocked dependencies to ensure
fast execution and isolation from other packages.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from tensorflow.keras.callbacks import EarlyStopping

# TODO remove these depenencies... no longer unit tests...
from img_classifier_config import BaseConfig
from img_classifier_data import BaseDataLoader
from img_classifier_models import BaseModel
from img_classifier_training import Trainer

pytestmark = pytest.mark.unit


class TestTrainer:
    """Unit tests for Trainer class with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures with mocked dependencies."""
        self.temp_dir = isolated_tmp_dir

        # Create mock config
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
        self.mock_config.early_stopping_patience = 5
        self.mock_config.accuracy_threshold = 0.995
        self.mock_config.models_dir = self.temp_dir / "models"
        self.mock_config.models_dir.mkdir(parents=True, exist_ok=True)

        # Create mock model
        self.mock_model = Mock()
        self.mock_keras_model = Mock()
        self.mock_model.model = self.mock_keras_model
        self.mock_model.save = Mock()

        # Create mock data loader
        self.mock_loader = Mock()

        # Import Trainer with mocked dependencies
        with patch("img_classifier_training.trainer.keras"):
            from img_classifier_training import Trainer

            self.Trainer = Trainer
            self.trainer = Trainer(self.mock_config, self.mock_model, self.mock_loader)

        yield

    def test_initialization(self):
        """Test Trainer initialization."""
        assert self.trainer.config == self.mock_config
        assert self.trainer.model == self.mock_model
        assert self.trainer.data_loader == self.mock_loader
        assert self.trainer.history is None

    def test_prepare_data_basic(self):
        """Test basic data preparation."""
        # Mock data
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

        result = self.trainer.prepare_data()

        assert len(result) == 6
        X_train_out, y_train_out, X_val_out, y_val_out, X_test_out, y_test_out = result

        # Check shapes
        assert X_train_out.shape == X_train.shape
        assert X_val_out.shape[0] == 20
        assert X_test_out.shape[0] == 20

        # Check labels are categorical
        assert y_train_out.shape == (100, 4)
        assert y_val_out.shape == (20, 4)
        assert y_test_out.shape == (20, 4)

    def test_prepare_data_with_reduction(self):
        """Test data preparation with dataset reduction."""
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

        # Mock reduction to return half the data
        self.mock_loader.reduce_dataset.side_effect = [
            (X_train[:50], y_train[:50]),  # train
            (X_test[:10], y_test[:10]),  # val
            (X_test[20:30], y_test[20:30]),  # test
        ]

        self.trainer.prepare_data()

        # Verify reduction was called
        assert self.mock_loader.reduce_dataset.call_count == 3

    def test_prepare_data_full_dataset(self):
        """Test data preparation with full dataset (data_percent=1.0)."""
        self.mock_config.data_percent = 1.0

        X_train = np.random.rand(50, 128, 128, 3)
        y_train = np.random.randint(0, 4, 50)
        X_test = np.random.rand(20, 128, 128, 3)
        y_test = np.random.randint(0, 4, 20)

        self.mock_loader.load_train_data.return_value = (X_train, y_train)
        self.mock_loader.load_test_data.return_value = (X_test, y_test)
        self.mock_loader.split_data.return_value = (
            X_test[:10],
            X_test[10:],
            y_test[:10],
            y_test[10:],
        )

        self.trainer.prepare_data()

        # Verify reduction was NOT called
        self.mock_loader.reduce_dataset.assert_not_called()

    def test_get_callbacks_all_enabled(self):
        """Test get_callbacks with all callbacks enabled."""
        # Mock callback classes
        mock_early_stopping = Mock()
        mock_checkpoint = Mock()
        mock_accuracy_callback = Mock()

        with patch(
            "img_classifier_training.trainer.EarlyStopping", return_value=mock_early_stopping
        ):
            with patch(
                "img_classifier_training.trainer.ModelCheckpoint", return_value=mock_checkpoint
            ):
                with patch(
                    "img_classifier_training.trainer.AccuracyThresholdCallback",
                    return_value=mock_accuracy_callback,
                ):
                    callbacks = self.trainer.get_callbacks()

        assert len(callbacks) == 3
        assert mock_early_stopping in callbacks
        assert mock_checkpoint in callbacks
        assert mock_accuracy_callback in callbacks

    def test_get_callbacks_early_stopping_only(self):
        """Test get_callbacks with only early stopping."""
        self.mock_config.use_early_stopping = True
        self.mock_config.use_model_checkpoint = False
        self.mock_config.use_accuracy_threshold_stopping = False

        mock_early_stopping = Mock()
        with patch(
            "img_classifier_training.trainer.EarlyStopping", return_value=mock_early_stopping
        ):
            callbacks = self.trainer.get_callbacks()

        assert len(callbacks) == 1
        assert mock_early_stopping in callbacks

    def test_get_callbacks_checkpoint_only(self):
        """Test get_callbacks with only model checkpoint."""
        self.mock_config.use_early_stopping = False
        self.mock_config.use_model_checkpoint = True
        self.mock_config.use_accuracy_threshold_stopping = False

        mock_checkpoint = Mock()
        with patch("img_classifier_training.trainer.ModelCheckpoint", return_value=mock_checkpoint):
            callbacks = self.trainer.get_callbacks()

        assert len(callbacks) == 1
        assert mock_checkpoint in callbacks

    def test_get_callbacks_accuracy_threshold_only(self):
        """Test get_callbacks with only accuracy threshold."""
        self.mock_config.use_early_stopping = False
        self.mock_config.use_model_checkpoint = False
        self.mock_config.use_accuracy_threshold_stopping = True

        mock_accuracy_callback = Mock()
        with patch(
            "img_classifier_training.trainer.AccuracyThresholdCallback",
            return_value=mock_accuracy_callback,
        ):
            callbacks = self.trainer.get_callbacks()

        assert len(callbacks) == 1
        assert mock_accuracy_callback in callbacks

    def test_get_callbacks_none_enabled(self):
        """Test get_callbacks with no callbacks enabled."""
        self.mock_config.use_early_stopping = False
        self.mock_config.use_model_checkpoint = False
        self.mock_config.use_accuracy_threshold_stopping = False

        callbacks = self.trainer.get_callbacks()

        assert len(callbacks) == 0

    @patch("img_classifier_training.trainer.time")
    def test_train_basic(self, mock_time):
        """Test basic training."""
        mock_time.time.side_effect = [1000.0, 1100.0]  # 100s training

        X_train = np.random.rand(10, 128, 128, 3)
        y_train = np.random.randint(0, 4, (10, 4))
        X_val = np.random.rand(5, 128, 128, 3)
        y_val = np.random.randint(0, 4, (5, 4))

        mock_history = Mock()
        self.mock_keras_model.fit.return_value = mock_history

        result = self.trainer.train(X_train, y_train, X_val, y_val)

        assert result == mock_history
        assert self.trainer.history == mock_history
        self.mock_keras_model.fit.assert_called_once()

    @patch("img_classifier_training.trainer.time")
    def test_train_compiles_if_needed(self, mock_time):
        """Test that train compiles model if not already compiled."""
        mock_time.time.side_effect = [1000.0, 1100.0]

        # Set model to None initially
        self.trainer.model.model = None

        # Set up compile to set the model
        def compile_side_effect():
            self.trainer.model.model = self.mock_keras_model

        self.mock_model.compile.side_effect = compile_side_effect

        X_train = np.random.rand(10, 128, 128, 3)
        y_train = np.random.randint(0, 4, (10, 4))
        X_val = np.random.rand(5, 128, 128, 3)
        y_val = np.random.randint(0, 4, (5, 4))

        mock_history = Mock()
        self.mock_keras_model.fit.return_value = mock_history

        self.trainer.train(X_train, y_train, X_val, y_val)

        self.mock_model.compile.assert_called_once()

    def test_train_with_callbacks(self):
        """Test training with callbacks."""
        X_train = np.random.rand(10, 128, 128, 3)
        y_train = np.random.randint(0, 4, (10, 4))
        X_val = np.random.rand(5, 128, 128, 3)
        y_val = np.random.randint(0, 4, (5, 4))

        self.config.use_early_stopping = True

        mock_history = Mock()
        self.mock_keras_model.fit.return_value = mock_history

        self.trainer.train(X_train, y_train, X_val, y_val)

        # Verify fit was called with callbacks
        call_args = self.mock_keras_model.fit.call_args
        assert "callbacks" in call_args.kwargs
        assert len(call_args.kwargs["callbacks"]) > 0

    def test_evaluate_basic(self):
        """Test model evaluation."""
        X_test = np.random.rand(10, 128, 128, 3)
        y_test = np.random.randint(0, 4, (10, 4))

        self.mock_keras_model.evaluate.return_value = (0.15, 0.95)

        loss, acc = self.trainer.evaluate(X_test, y_test)

        assert loss == 0.15
        assert acc == 0.95
        self.mock_keras_model.evaluate.assert_called_once_with(X_test, y_test, verbose=0)

    def test_evaluate_no_model(self):
        """Test evaluate raises error when model is not built."""
        self.mock_model.model = None

        X_test = np.random.rand(10, 128, 128, 3)
        y_test = np.random.randint(0, 4, (10, 4))

        with pytest.raises(AssertionError, match="Model must be built before evaluation"):
            self.trainer.evaluate(X_test, y_test)

    @patch("img_classifier_training.trainer.plt")
    @patch("img_classifier_training.trainer.pd")
    @patch("img_classifier_training.trainer.sns")
    def test_plot_history_with_history(self, mock_sns, mock_pd, mock_plt):
        """Test plotting training history."""
        mock_history = Mock()
        mock_history.history = {
            "loss": [0.5, 0.3, 0.2],
            "acc": [0.7, 0.85, 0.9],
            "val_loss": [0.6, 0.35, 0.25],
            "val_acc": [0.65, 0.8, 0.88],
        }
        self.trainer.history = mock_history

        mock_df = Mock()
        mock_pd.DataFrame.return_value = mock_df
        mock_df.rename_axis.return_value = mock_df
        mock_df.reset_index.return_value = mock_df
        mock_df.melt.return_value = mock_df
        mock_df.__getitem__ = Mock(return_value=mock_df)

        # Mock subplots to return figure and axes
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        self.trainer.plot_history()

        mock_plt.show.assert_called_once()

    @patch("img_classifier_training.trainer.plt")
    @patch("img_classifier_training.trainer.pd.DataFrame")
    @patch("img_classifier_training.trainer.sns")
    def test_plot_history_with_save_path(self, mock_sns, mock_dataframe, mock_plt):
        """Test plotting with save path."""
        mock_history = Mock()
        mock_history.history = {"loss": [0.5], "acc": [0.8], "val_loss": [0.6], "val_acc": [0.75]}
        self.trainer.history = mock_history

        # Mock DataFrame
        mock_df = Mock()
        mock_dataframe.return_value = mock_df
        mock_df.rename_axis.return_value = mock_df
        mock_df.reset_index.return_value = mock_df
        mock_df.melt.return_value = mock_df
        mock_df.__getitem__ = Mock(return_value=mock_df)

        # Mock subplots
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        save_path = self.temp_dir / "plot.png"
        self.trainer.plot_history(save_path=save_path)

        mock_plt.savefig.assert_called_once_with(save_path)

    def test_plot_history_no_history(self):
        """Test plotting when no history is available."""
        self.trainer.history = None

        # Should not raise an error
        self.trainer.plot_history()

    def test_save_model_above_threshold(self):
        """Test model saving when accuracy is above threshold."""
        self.config.min_accuracy_to_save = 0.95
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        result = self.trainer.save_model(acc=0.97, loss=0.1, elapsed_time=100.0)

        assert result is not None
        assert "97%_acc" in result.name
        self.mock_model.save.assert_called_once()

    def test_save_model_below_threshold(self):
        """Test model not saved when accuracy is below threshold."""
        self.config.min_accuracy_to_save = 0.95

        result = self.trainer.save_model(acc=0.90, loss=0.2, elapsed_time=100.0)

        assert result is None
        self.mock_model.save.assert_not_called()

    def test_save_model_force_save(self):
        """Test model saved with force_save even below threshold."""
        self.config.min_accuracy_to_save = 0.95
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        result = self.trainer.save_model(acc=0.85, loss=0.3, elapsed_time=100.0, force_save=True)

        assert result is not None
        assert "85%_acc" in result.name
        self.mock_model.save.assert_called_once()

    def test_save_model_filename_format(self):
        """Test model filename format."""
        self.config.min_accuracy_to_save = 0.0
        self.config.project_name = "test_project"
        self.config.num_epochs = 25
        self.config.batch_size = 32
        self.config.learning_rate = 0.001
        self.config.data_percent = 0.8
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        result = self.trainer.save_model(acc=0.96, loss=0.12, elapsed_time=120.5)

        assert result is not None
        filename = result.name
        assert "test_project" in filename
        assert "96%_acc" in filename
        assert "25_ep" in filename
        assert "32_bs" in filename
        assert "0.001_lr" in filename
        assert "80%_data" in filename
        assert "0.12_loss" in filename
        assert "120s" in filename

    def test_log_results_creates_log_file(self):
        """Test log_results creates log file."""
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.config.logs_dir / "training_history.log"

        self.trainer.log_results(acc=0.95, loss=0.15, elapsed_time=100.0)

        assert log_file.exists()
        content = log_file.read_text()
        assert "Training Run:" in content
        assert "accuracy,loss,data_percent" in content
        assert "0.9500" in content
        assert "0.1500" in content

    def test_log_results_appends_to_existing(self):
        """Test log_results appends to existing log file."""
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.config.logs_dir / "training_history.log"

        # First log
        self.trainer.log_results(acc=0.90, loss=0.20, elapsed_time=50.0)

        # Second log
        self.trainer.log_results(acc=0.95, loss=0.15, elapsed_time=60.0)

        content = log_file.read_text()
        lines = content.split("\n")

        # Should have header + 2 data lines
        assert len([l for l in lines if l and not l.startswith("Training")]) >= 3

    def test_log_results_format(self):
        """Test log results format."""
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.data_percent = 0.75
        self.config.batch_size = 64
        self.config.learning_rate = 0.002
        self.config.num_epochs = 30

        self.trainer.log_results(acc=0.92, loss=0.18, elapsed_time=75.0)

        log_file = self.config.logs_dir / "training_history.log"
        content = log_file.read_text()

        assert "0.9200" in content
        assert "0.1800" in content
        assert "0.75" in content
        assert "64" in content
        assert "0.002" in content
        assert "30" in content
        assert "75" in content

    @patch.object(Trainer, "prepare_data")
    @patch.object(Trainer, "train")
    @patch.object(Trainer, "evaluate")
    @patch.object(Trainer, "save_model")
    @patch.object(Trainer, "log_results")
    @patch.object(Trainer, "cleanup")
    def test_run_complete_pipeline(
        self, mock_cleanup, mock_log, mock_save, mock_eval, mock_train, mock_prepare
    ):
        """Test complete run pipeline."""
        # Setup mocks
        mock_prepare.return_value = (
            np.array([1]),
            np.array([0]),
            np.array([2]),
            np.array([1]),
            np.array([3]),
            np.array([0]),
        )
        mock_history = Mock()
        mock_train.return_value = mock_history
        mock_eval.return_value = (0.15, 0.95)
        mock_save.return_value = Path("/tmp/model.keras")

        # Run
        acc, loss = self.trainer.run(plot=False, force_save=True)

        # Verify
        assert acc == 0.95
        assert loss == 0.15
        mock_prepare.assert_called_once()
        mock_train.assert_called_once()
        mock_eval.assert_called_once()
        mock_save.assert_called_once()
        mock_log.assert_called_once()
        mock_cleanup.assert_called_once()

    @patch.object(Trainer, "prepare_data")
    @patch.object(Trainer, "plot_history")
    @patch.object(Trainer, "cleanup")
    def test_run_with_plot(self, mock_cleanup, mock_plot, mock_prepare):
        """Test run with plotting enabled."""
        mock_prepare.return_value = (
            np.random.rand(10, 128, 128, 3),
            np.zeros((10, 4)),
            np.random.rand(5, 128, 128, 3),
            np.zeros((5, 4)),
            np.random.rand(5, 128, 128, 3),
            np.zeros((5, 4)),
        )
        self.mock_keras_model.fit.return_value = Mock()
        self.mock_keras_model.evaluate.return_value = (0.1, 0.96)
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        self.trainer.run(plot=True)

        mock_plot.assert_called_once()

    @patch.object(Trainer, "cleanup")
    def test_run_cleanup_on_exception(self, mock_cleanup):
        """Test that cleanup is called even on exception."""
        self.mock_loader.load_train_data.side_effect = Exception("Data load failed")

        with pytest.raises(Exception):
            self.trainer.run()

        mock_cleanup.assert_called_once()

    @patch("img_classifier_training.trainer.K")
    @patch("img_classifier_training.trainer.gc")
    def test_cleanup_clears_model(self, mock_gc, mock_K):
        """Test cleanup clears model and resources."""
        self.trainer.cleanup()

        assert self.mock_model.model is None
        mock_K.clear_session.assert_called_once()
        mock_gc.collect.assert_called_once()

    @patch("img_classifier_training.trainer.K")
    @patch("img_classifier_training.trainer.gc")
    def test_cleanup_no_model(self, mock_gc, mock_K):
        """Test cleanup when model is None."""
        self.mock_model.model = None

        # Should not raise an error
        self.trainer.cleanup()

        mock_K.clear_session.assert_called_once()
        mock_gc.collect.assert_called_once()


@pytest.mark.unit
class TestTrainerEdgeCases:
    """Edge case tests for Trainer."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = BaseConfig(working_dir=self.temp_dir)
        self.mock_model = Mock(spec=BaseModel)
        self.mock_model.model = Mock()
        self.mock_loader = Mock(spec=BaseDataLoader)
        self.trainer = Trainer(self.config, self.mock_model, self.mock_loader)
        yield

    def test_zero_epochs(self):
        """Test with zero epochs."""
        self.config.num_epochs = 0

        X_train = np.random.rand(10, 128, 128, 3)
        y_train = np.random.randint(0, 4, (10, 4))
        X_val = np.random.rand(5, 128, 128, 3)
        y_val = np.random.randint(0, 4, (5, 4))

        mock_history = Mock()
        self.mock_model.model.fit.return_value = mock_history

        self.trainer.train(X_train, y_train, X_val, y_val)

        # Should still call fit with 0 epochs
        call_args = self.mock_model.model.fit.call_args
        assert call_args.kwargs["epochs"] == 0

    def test_single_batch_size(self):
        """Test with batch size of 1."""
        self.config.batch_size = 1

        X_train = np.random.rand(10, 128, 128, 3)
        y_train = np.random.randint(0, 4, (10, 4))
        X_val = np.random.rand(5, 128, 128, 3)
        y_val = np.random.randint(0, 4, (5, 4))

        mock_history = Mock()
        self.mock_model.model.fit.return_value = mock_history

        self.trainer.train(X_train, y_train, X_val, y_val)

        call_args = self.mock_model.model.fit.call_args
        assert call_args.kwargs["batch_size"] == 1

    def test_large_batch_size(self):
        """Test with very large batch size."""
        self.config.batch_size = 10000

        X_train = np.random.rand(100, 128, 128, 3)
        y_train = np.random.randint(0, 4, (100, 4))
        X_val = np.random.rand(20, 128, 128, 3)
        y_val = np.random.randint(0, 4, (20, 4))

        mock_history = Mock()
        self.mock_model.model.fit.return_value = mock_history

        self.trainer.train(X_train, y_train, X_val, y_val)

        call_args = self.mock_model.model.fit.call_args
        assert call_args.kwargs["batch_size"] == 10000

    def test_empty_training_data(self):
        """Test with empty training data."""
        X_train = np.array([]).reshape(0, 128, 128, 3)
        y_train = np.array([]).reshape(0, 4)
        X_val = np.random.rand(5, 128, 128, 3)
        y_val = np.random.randint(0, 4, (5, 4))

        mock_history = Mock()
        self.mock_model.model.fit.return_value = mock_history

        # Should not raise, just attempt to train
        self.trainer.train(X_train, y_train, X_val, y_val)

    def test_perfect_accuracy(self):
        """Test saving model with 100% accuracy."""
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        result = self.trainer.save_model(acc=1.0, loss=0.0, elapsed_time=100.0)

        assert result is not None
        assert "100%_acc" in result.name

    def test_zero_accuracy(self):
        """Test with 0% accuracy."""
        self.config.min_accuracy_to_save = 0.0
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        result = self.trainer.save_model(acc=0.0, loss=999.0, elapsed_time=1.0)

        assert result is not None
        assert "0%_acc" in result.name

    def test_very_long_training_time(self):
        """Test with very long training time."""
        self.config.min_accuracy_to_save = 0.0  # Allow saving
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        result = self.trainer.save_model(acc=0.95, loss=0.1, elapsed_time=999999.0)

        assert result is not None
        assert "999999s" in result.name

    def test_custom_early_stopping_patience(self):
        """Test custom early stopping patience."""
        self.config.use_early_stopping = True
        self.config.early_stopping_patience = 5

        callbacks = self.trainer.get_callbacks()

        early_stopping = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        assert early_stopping.patience == 5

    def test_extreme_learning_rate(self):
        """Test with extreme learning rate in filename."""
        self.config.learning_rate = 0.0000001
        self.config.min_accuracy_to_save = 0.0  # Allow saving
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        result = self.trainer.save_model(acc=0.95, loss=0.1, elapsed_time=100.0)

        assert result is not None
        assert "1e-07_lr" in result.name or "0.0000001_lr" in result.name
