"""Unit tests for the CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from img_classifier_cli.main import cli, main


pytestmark = pytest.mark.unit


@pytest.fixture
def runner():
    """Provide a Click test runner."""
    return CliRunner()


@pytest.fixture
def dataset_dir(tmp_path):
    """Create a minimal fake dataset directory so Path(exists=True) passes."""
    d = tmp_path / "dataset"
    d.mkdir()
    return d


@pytest.fixture
def model_file(tmp_path):
    """Create a fake model file."""
    f = tmp_path / "model.keras"
    f.write_bytes(b"fake")
    return f


@pytest.fixture
def image_file(tmp_path):
    """Create a fake image file."""
    f = tmp_path / "image.jpg"
    f.write_bytes(b"fake")
    return f


# ---------------------------------------------------------------------------
# cli group
# ---------------------------------------------------------------------------


class TestCliGroup:
    """Tests for the root cli group."""

    def test_help(self, runner: CliRunner):
        """--help should exit 0 and print usage."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_version(self, runner: CliRunner):
        """--version should exit 0 and print version string."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_no_args_shows_usage(self, runner: CliRunner):
        """Invoking without subcommand should show usage (exit 0 or 2 depending on Click)."""
        result = runner.invoke(cli, [])
        # Click groups show a "Missing command." error by default (exit 0 with standalone_mode or 2)
        assert "Usage" in result.output or result.exit_code in (0, 2)


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


class TestInfoCommand:
    """Tests for the `info` subcommand."""

    def test_info_success(self, runner: CliRunner, dataset_dir: Path):
        """info should display dataset statistics."""
        mock_info = {
            "dataset_path": str(dataset_dir),
            "num_classes": 3,
            "class_names": ["cat", "dog", "fish"],
            "total_images": 300,
            "class_distribution": {"cat": 100, "dog": 100, "fish": 100},
            "is_balanced": True,
            "has_train_test_split": False,
            "recommended_complexity": "simple",
            "sample_image_shape": (224, 224, 3),
        }

        with patch("img_classifier_cli.main.DatasetDetector") as MockDetector:
            MockDetector.return_value.detect.return_value = mock_info
            result = runner.invoke(cli, ["info", str(dataset_dir)])

        assert result.exit_code == 0
        assert "3" in result.output  # num_classes
        assert "cat" in result.output

    def test_info_no_sample_shape(self, runner: CliRunner, dataset_dir: Path):
        """info should handle None sample_image_shape gracefully."""
        mock_info = {
            "dataset_path": str(dataset_dir),
            "num_classes": 2,
            "class_names": ["a", "b"],
            "total_images": 10,
            "class_distribution": {"a": 5, "b": 5},
            "is_balanced": True,
            "has_train_test_split": False,
            "recommended_complexity": "simple",
            "sample_image_shape": None,
        }

        with patch("img_classifier_cli.main.DatasetDetector") as MockDetector:
            MockDetector.return_value.detect.return_value = mock_info
            result = runner.invoke(cli, ["info", str(dataset_dir)])

        assert result.exit_code == 0

    def test_info_invalid_path(self, runner: CliRunner, tmp_path: Path):
        """info with a non-existent path should fail."""
        result = runner.invoke(cli, ["info", str(tmp_path / "nonexistent")])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------


class TestTrainCommand:
    """Tests for the `train` subcommand."""

    def test_train_success(self, runner: CliRunner, dataset_dir: Path):
        """train should call orchestrator.run and report success."""
        with (
            patch("img_classifier_cli.main.DatasetDetector") as MockDetector,
            patch("img_classifier_cli.main.TrainingOrchestrator") as MockOrchestrator,
        ):
            mock_cfg = MagicMock()
            MockDetector.return_value.create_config.return_value = mock_cfg
            MockOrchestrator.return_value.run.return_value = {}

            result = runner.invoke(cli, ["train", str(dataset_dir)])

        assert result.exit_code == 0
        assert "completed" in result.output.lower()

    def test_train_with_config_file(self, runner: CliRunner, dataset_dir: Path, tmp_path: Path):
        """train --config should load config from file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("project_name: test\n")

        with (
            patch("img_classifier_cli.main.DatasetConfig") as MockConfig,
            patch("img_classifier_cli.main.TrainingOrchestrator") as MockOrchestrator,
        ):
            mock_cfg = MagicMock()
            MockConfig.from_yaml.return_value = mock_cfg
            MockOrchestrator.return_value.run.return_value = {}

            result = runner.invoke(cli, ["train", str(dataset_dir), "--config", str(config_file)])

        assert result.exit_code == 0
        MockConfig.from_yaml.assert_called_once()

    def test_train_keyboard_interrupt(self, runner: CliRunner, dataset_dir: Path):
        """train should handle KeyboardInterrupt and exit with code 1."""
        with (
            patch("img_classifier_cli.main.DatasetDetector") as MockDetector,
            patch("img_classifier_cli.main.TrainingOrchestrator") as MockOrchestrator,
        ):
            mock_cfg = MagicMock()
            MockDetector.return_value.create_config.return_value = mock_cfg
            MockOrchestrator.return_value.run.side_effect = KeyboardInterrupt

            result = runner.invoke(cli, ["train", str(dataset_dir)])

        assert result.exit_code == 1

    def test_train_exception(self, runner: CliRunner, dataset_dir: Path):
        """train should handle unexpected exceptions and exit with code 1."""
        with (
            patch("img_classifier_cli.main.DatasetDetector") as MockDetector,
            patch("img_classifier_cli.main.TrainingOrchestrator") as MockOrchestrator,
        ):
            mock_cfg = MagicMock()
            MockDetector.return_value.create_config.return_value = mock_cfg
            MockOrchestrator.return_value.run.side_effect = RuntimeError("oops")

            result = runner.invoke(cli, ["train", str(dataset_dir)])

        assert result.exit_code == 1

    def test_train_epoch_override(self, runner: CliRunner, dataset_dir: Path):
        """--epochs flag should override config epochs."""
        with (
            patch("img_classifier_cli.main.DatasetDetector") as MockDetector,
            patch("img_classifier_cli.main.TrainingOrchestrator") as MockOrchestrator,
        ):
            mock_cfg = MagicMock()
            MockDetector.return_value.create_config.return_value = mock_cfg
            MockOrchestrator.return_value.run.return_value = {}

            runner.invoke(cli, ["train", str(dataset_dir), "--epochs", "5"])

        assert mock_cfg.num_epochs == 5


# ---------------------------------------------------------------------------
# optimize command
# ---------------------------------------------------------------------------


class TestOptimizeCommand:
    """Tests for the `optimize` subcommand."""

    def test_optimize_success(self, runner: CliRunner, dataset_dir: Path):
        """optimize should run hyperparameter optimization and report success."""
        with (
            patch("img_classifier_cli.main.DatasetDetector") as MockDetector,
            patch("img_classifier_cli.main.TrainingOrchestrator") as MockOrchestrator,
        ):
            mock_cfg = MagicMock()
            MockDetector.return_value.create_config.return_value = mock_cfg
            mock_orch = MockOrchestrator.return_value
            mock_orch.run.return_value = {}
            mock_orch.optimizer = None  # No optimizer to display

            result = runner.invoke(cli, ["optimize", str(dataset_dir)])

        assert result.exit_code == 0
        assert "completed" in result.output.lower()

    def test_optimize_with_summary(self, runner: CliRunner, dataset_dir: Path):
        """optimize should display optimizer summary when available."""
        with (
            patch("img_classifier_cli.main.DatasetDetector") as MockDetector,
            patch("img_classifier_cli.main.TrainingOrchestrator") as MockOrchestrator,
        ):
            mock_cfg = MagicMock()
            MockDetector.return_value.create_config.return_value = mock_cfg
            mock_orch = MockOrchestrator.return_value
            mock_orch.run.return_value = {}
            mock_opt = MagicMock()
            mock_opt.get_summary.return_value = "Best val_accuracy: 0.95"
            mock_orch.optimizer = mock_opt

            result = runner.invoke(cli, ["optimize", str(dataset_dir)])

        assert result.exit_code == 0
        assert "0.95" in result.output

    def test_optimize_keyboard_interrupt_shows_partial(self, runner: CliRunner, dataset_dir: Path):
        """optimize should show partial results on KeyboardInterrupt."""
        with (
            patch("img_classifier_cli.main.DatasetDetector") as MockDetector,
            patch("img_classifier_cli.main.TrainingOrchestrator") as MockOrchestrator,
        ):
            mock_cfg = MagicMock()
            MockDetector.return_value.create_config.return_value = mock_cfg
            mock_orch = MockOrchestrator.return_value
            mock_orch.run.side_effect = KeyboardInterrupt
            mock_opt = MagicMock()
            mock_opt.results = [MagicMock()]
            mock_opt.get_summary.return_value = "Partial"
            mock_orch.optimizer = mock_opt

            result = runner.invoke(cli, ["optimize", str(dataset_dir)])

        assert result.exit_code == 1

    def test_optimize_quick_flag(self, runner: CliRunner, dataset_dir: Path):
        """--quick flag should use quick search space."""
        with (
            patch("img_classifier_cli.main.DatasetDetector") as MockDetector,
            patch("img_classifier_cli.main.TrainingOrchestrator") as MockOrchestrator,
            patch("img_classifier_cli.main.HyperparameterSpace") as MockSpace,
        ):
            mock_cfg = MagicMock()
            MockDetector.return_value.create_config.return_value = mock_cfg
            MockOrchestrator.return_value.run.return_value = {}
            MockOrchestrator.return_value.optimizer = None

            runner.invoke(cli, ["optimize", str(dataset_dir), "--quick"])

        MockSpace.quick.assert_called_once()


# ---------------------------------------------------------------------------
# create-config command
# ---------------------------------------------------------------------------


class TestCreateConfigCommand:
    """Tests for the `create-config` subcommand."""

    def test_create_config_success(self, runner: CliRunner, dataset_dir: Path, tmp_path: Path):
        """create-config should write config to output path."""
        output = tmp_path / "config.yaml"

        with patch("img_classifier_cli.main.DatasetDetector") as MockDetector:
            mock_cfg = MagicMock()
            MockDetector.return_value.create_config.return_value = mock_cfg

            result = runner.invoke(cli, ["create-config", str(dataset_dir), str(output)])

        assert result.exit_code == 0
        mock_cfg.to_yaml.assert_called_once_with(output)
        assert str(output) in result.output

    def test_create_config_invalid_dataset(self, runner: CliRunner, tmp_path: Path):
        """create-config with a non-existent dataset path should fail."""
        result = runner.invoke(
            cli, ["create-config", str(tmp_path / "none"), str(tmp_path / "out.yaml")]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# predict command
# ---------------------------------------------------------------------------


class TestPredictCommand:
    """Tests for the `predict` subcommand."""

    def test_predict_success(self, runner: CliRunner, model_file: Path, image_file: Path):
        """predict should output predicted class and confidence."""
        import numpy as np

        fake_preds = np.array([[0.1, 0.7, 0.2]])

        mock_model = MagicMock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.predict.return_value = fake_preds

        fake_image = np.zeros((224, 224, 3), dtype=np.uint8)

        # predict imports tf/cv2/np lazily inside the function body, so patch their modules
        with (
            patch("tensorflow.keras.models.load_model", return_value=mock_model),
            patch("cv2.imread", return_value=fake_image),
            patch("cv2.resize", return_value=fake_image),
        ):
            result = runner.invoke(
                cli,
                [
                    "predict",
                    str(model_file),
                    str(image_file),
                    "--class-names",
                    "cat",
                    "--class-names",
                    "dog",
                    "--class-names",
                    "fish",
                ],
            )

        assert result.exit_code == 0
        assert "dog" in result.output  # highest probability (0.7)

    def test_predict_image_load_failure(
        self, runner: CliRunner, model_file: Path, image_file: Path
    ):
        """predict should exit 1 when cv2 cannot decode the image."""
        mock_model = MagicMock()
        mock_model.input_shape = (None, 224, 224, 3)

        with (
            patch("tensorflow.keras.models.load_model", return_value=mock_model),
            patch("cv2.imread", return_value=None),  # Simulate decode failure
        ):
            result = runner.invoke(cli, ["predict", str(model_file), str(image_file)])

        assert result.exit_code == 1

    def test_predict_no_class_names(self, runner: CliRunner, model_file: Path, image_file: Path):
        """predict without --class-names should show class index."""
        import numpy as np

        fake_preds = np.array([[0.3, 0.7]])
        mock_model = MagicMock()
        mock_model.input_shape = (None, 64, 64, 3)
        mock_model.predict.return_value = fake_preds
        fake_image = np.zeros((64, 64, 3), dtype=np.uint8)

        with (
            patch("tensorflow.keras.models.load_model", return_value=mock_model),
            patch("cv2.imread", return_value=fake_image),
            patch("cv2.resize", return_value=fake_image),
        ):
            result = runner.invoke(cli, ["predict", str(model_file), str(image_file)])

        assert result.exit_code == 0
        assert "index" in result.output.lower() or "1" in result.output


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() entry point."""

    def test_main_delegates_to_cli(self, runner: CliRunner):
        """main() should call cli() which shows help with no args."""
        with patch("img_classifier_cli.main.cli") as mock_cli:
            main()
            mock_cli.assert_called_once()
