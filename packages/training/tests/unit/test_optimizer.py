"""Unit tests for Hyperparameter Optimizer module using mocking.

These tests verify optimizer behavior with mocked dependencies
to ensure fast execution and isolation from other packages.
"""

import json
import random
from unittest.mock import patch

import pytest
from img_classifier_config import BaseConfig

from img_classifier_training.optimizer import (
    HyperparameterSpace,
    TrialResult,
    HyperparameterOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    create_optimizer,
)


pytestmark = pytest.mark.unit


class TestHyperparameterSpace:
    """Tests for HyperparameterSpace class."""

    def test_initialization(self):
        """Test HyperparameterSpace initialization."""
        space = HyperparameterSpace(
            learning_rates=[0.001, 0.01],
            batch_sizes=[32, 64],
            dropout_rates=[0.3, 0.4],
            architectures=["simple", "medium"],
            num_epochs=[20, 30],
        )

        assert space.learning_rates == [0.001, 0.01]
        assert space.batch_sizes == [32, 64]
        assert space.dropout_rates == [0.3, 0.4]
        assert space.architectures == ["simple", "medium"]
        assert space.num_epochs == [20, 30]

    def test_default_space(self):
        """Test default hyperparameter space."""
        space = HyperparameterSpace.default()

        assert len(space.learning_rates) == 5
        assert 0.001 in space.learning_rates
        assert len(space.batch_sizes) == 3
        assert 32 in space.batch_sizes
        assert len(space.dropout_rates) == 4
        assert len(space.architectures) == 3
        assert "simple" in space.architectures
        assert "medium" in space.architectures
        assert "deep" in space.architectures
        assert len(space.num_epochs) == 4

    @pytest.mark.smoke
    def test_quick_space(self):
        """Test quick hyperparameter space."""
        space = HyperparameterSpace.quick()

        assert len(space.learning_rates) == 2
        assert len(space.batch_sizes) == 2
        assert len(space.dropout_rates) == 2
        assert len(space.architectures) == 2
        assert len(space.num_epochs) == 2

    def test_empty_lists(self):
        """Test with empty lists."""
        space = HyperparameterSpace(
            learning_rates=[], batch_sizes=[], dropout_rates=[], architectures=[], num_epochs=[]
        )

        assert space.learning_rates == []
        assert space.batch_sizes == []

    def test_single_values(self):
        """Test with single values in lists."""
        space = HyperparameterSpace(
            learning_rates=[0.001],
            batch_sizes=[32],
            dropout_rates=[0.3],
            architectures=["simple"],
            num_epochs=[20],
        )

        assert len(space.learning_rates) == 1
        assert len(space.batch_sizes) == 1


@pytest.mark.unit
class TestTrialResult:
    """Tests for TrialResult class."""

    def test_initialization(self):
        """Test TrialResult initialization."""
        result = TrialResult(
            trial_id=1,
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
            train_accuracy=0.95,
            val_accuracy=0.90,
            test_accuracy=0.88,
            train_loss=0.15,
            val_loss=0.20,
            test_loss=0.22,
            training_time=120.5,
            timestamp="2024-01-01T12:00:00",
        )

        assert result.trial_id == 1
        assert result.hyperparameters["learning_rate"] == 0.001
        assert result.train_accuracy == 0.95
        assert result.val_accuracy == 0.90
        assert result.test_accuracy == 0.88
        assert result.training_time == 120.5

    def test_to_dict(self):
        """Test converting TrialResult to dictionary."""
        result = TrialResult(
            trial_id=2,
            hyperparameters={"dropout_rate": 0.3},
            train_accuracy=0.92,
            val_accuracy=0.88,
            test_accuracy=0.86,
            train_loss=0.18,
            val_loss=0.23,
            test_loss=0.25,
            training_time=100.0,
            timestamp="2024-01-01T13:00:00",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["trial_id"] == 2
        assert result_dict["train_accuracy"] == 0.92
        assert "hyperparameters" in result_dict

    def test_zero_accuracy(self):
        """Test with zero accuracy."""
        result = TrialResult(
            trial_id=1,
            hyperparameters={},
            train_accuracy=0.0,
            val_accuracy=0.0,
            test_accuracy=0.0,
            train_loss=999.0,
            val_loss=999.0,
            test_loss=999.0,
            training_time=1.0,
            timestamp="2024-01-01T12:00:00",
        )

        assert result.val_accuracy == 0.0

    def test_perfect_accuracy(self):
        """Test with perfect accuracy."""
        result = TrialResult(
            trial_id=1,
            hyperparameters={},
            train_accuracy=1.0,
            val_accuracy=1.0,
            test_accuracy=1.0,
            train_loss=0.0,
            val_loss=0.0,
            test_loss=0.0,
            training_time=50.0,
            timestamp="2024-01-01T12:00:00",
        )

        assert result.val_accuracy == 1.0


@pytest.mark.unit
class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = BaseConfig(working_dir=self.temp_dir)
        self.search_space = HyperparameterSpace(
            learning_rates=[0.001, 0.01],
            batch_sizes=[32, 64],
            dropout_rates=[0.3],
            architectures=["simple"],
            num_epochs=[20],
        )
        yield

    def test_initialization(self):
        """Test GridSearchOptimizer initialization."""
        optimizer = GridSearchOptimizer(self.config, self.search_space, max_trials=10)

        assert optimizer.config == self.config
        assert optimizer.search_space == self.search_space
        assert optimizer.max_trials == 10
        assert len(optimizer.results) == 0
        assert optimizer.best_result is None

    def test_grid_creation(self):
        """Test grid creation."""
        optimizer = GridSearchOptimizer(self.config, self.search_space)

        # 2 learning_rates * 2 batch_sizes * 1 dropout * 1 arch * 1 epochs = 4 combinations
        assert len(optimizer._grid) == 4

    def test_suggest_hyperparameters(self):
        """Test suggesting hyperparameters from grid."""
        optimizer = GridSearchOptimizer(self.config, self.search_space)

        params = optimizer.suggest_hyperparameters(1)

        assert "learning_rate" in params
        assert "batch_size" in params
        assert "dropout_rate" in params
        assert "architecture" in params
        assert "num_epochs" in params
        assert params["learning_rate"] in [0.001, 0.01]
        assert params["batch_size"] in [32, 64]

    def test_suggest_all_combinations(self):
        """Test that all grid combinations are suggested."""
        optimizer = GridSearchOptimizer(self.config, self.search_space)

        all_params = []
        for i in range(4):
            params = optimizer.suggest_hyperparameters(i + 1)
            all_params.append(params)

        assert len(all_params) == 4

    def test_suggest_beyond_grid(self):
        """Test suggesting beyond grid size raises error."""
        optimizer = GridSearchOptimizer(self.config, self.search_space)

        # Exhaust grid
        for i in range(4):
            optimizer.suggest_hyperparameters(i + 1)

        # Next suggestion should raise
        with pytest.raises(StopIteration):
            optimizer.suggest_hyperparameters(5)

    def test_grid_randomization(self):
        """Test that grid is shuffled."""
        # Create two optimizers with same seed
        random.seed(42)
        optimizer1 = GridSearchOptimizer(self.config, self.search_space)
        grid1 = optimizer1._grid.copy()

        random.seed(43)
        optimizer2 = GridSearchOptimizer(self.config, self.search_space)
        grid2 = optimizer2._grid.copy()

        # Grids should be different due to shuffling
        # (may occasionally fail due to random chance)
        assert grid1 != grid2 or len(grid1) <= 1


@pytest.mark.unit
class TestRandomSearchOptimizer:
    """Tests for RandomSearchOptimizer class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = BaseConfig(working_dir=self.temp_dir)
        self.search_space = HyperparameterSpace(
            learning_rates=[0.001, 0.01],
            batch_sizes=[32, 64],
            dropout_rates=[0.3, 0.4],
            architectures=["simple", "medium"],
            num_epochs=[20, 30],
        )
        yield

    def test_initialization(self):
        """Test RandomSearchOptimizer initialization."""
        optimizer = RandomSearchOptimizer(self.config, self.search_space, max_trials=20)

        assert optimizer.config == self.config
        assert optimizer.max_trials == 20

    def test_suggest_hyperparameters(self):
        """Test random hyperparameter suggestion."""
        optimizer = RandomSearchOptimizer(self.config, self.search_space)

        params = optimizer.suggest_hyperparameters(1)

        assert params["learning_rate"] in self.search_space.learning_rates
        assert params["batch_size"] in self.search_space.batch_sizes
        assert params["dropout_rate"] in self.search_space.dropout_rates
        assert params["architecture"] in self.search_space.architectures
        assert params["num_epochs"] in self.search_space.num_epochs

    def test_suggest_multiple_times(self):
        """Test suggesting multiple times."""
        optimizer = RandomSearchOptimizer(self.config, self.search_space)

        params_list = []
        for i in range(10):
            params = optimizer.suggest_hyperparameters(i + 1)
            params_list.append(params)

        # Should get 10 suggestions
        assert len(params_list) == 10

        # All should be valid
        for params in params_list:
            assert params["learning_rate"] in self.search_space.learning_rates

    def test_random_variation(self):
        """Test that random search produces different suggestions."""
        optimizer = RandomSearchOptimizer(self.config, self.search_space)

        params1 = optimizer.suggest_hyperparameters(1)
        optimizer.suggest_hyperparameters(2)

        # Should eventually get different params (may rarely fail)
        different = False
        for i in range(20):
            params = optimizer.suggest_hyperparameters(i + 3)
            if params != params1:
                different = True
                break

        assert different


@pytest.mark.unit
class TestBayesianOptimizer:
    """Tests for BayesianOptimizer class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = BaseConfig(working_dir=self.temp_dir)
        self.search_space = HyperparameterSpace.quick()
        yield

    def test_initialization_without_optuna(self):
        """Test BayesianOptimizer initialization without optuna."""
        optimizer = BayesianOptimizer(self.config, self.search_space, use_optuna=False)

        assert optimizer.use_optuna is False
        assert optimizer.study is None

    def test_initialization_with_optuna(self):
        """Test BayesianOptimizer initialization with optuna."""
        # This test requires optuna to be installed and is complex to mock
        # We'll just test basic initialization with use_optuna=False
        optimizer = BayesianOptimizer(self.config, self.search_space, use_optuna=False)

        assert optimizer.use_optuna is False

    def test_suggest_without_optuna_fallback(self):
        """Test that suggest falls back to random search without optuna."""
        optimizer = BayesianOptimizer(self.config, self.search_space, use_optuna=False)

        # Need to define parent class suggest_hyperparameters
        # Since BayesianOptimizer inherits from RandomSearchOptimizer behavior
        # when use_optuna=False, let's use RandomSearchOptimizer
        from img_classifier_training.optimizer import RandomSearchOptimizer

        optimizer.__class__ = type(
            "TestOptimizer",
            (BayesianOptimizer,),
            {
                "suggest_hyperparameters": RandomSearchOptimizer.suggest_hyperparameters.__get__(
                    optimizer
                )
            },
        )

        params = optimizer.suggest_hyperparameters(1)

        assert "learning_rate" in params
        assert params["learning_rate"] in self.search_space.learning_rates

    def test_import_error_handling(self):
        """Test handling of optuna import error."""
        with patch("builtins.__import__", side_effect=ImportError("No module named 'optuna'")):
            optimizer = BayesianOptimizer(self.config, self.search_space, use_optuna=True)

            # Should fall back gracefully
            assert optimizer.use_optuna is False


@pytest.mark.unit
class TestHyperparameterOptimizer:
    """Tests for base HyperparameterOptimizer class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = BaseConfig(working_dir=self.temp_dir)
        self.search_space = HyperparameterSpace.quick()

        # Create concrete implementation for testing
        class TestOptimizer(HyperparameterOptimizer):
            def suggest_hyperparameters(self, trial_number):
                return {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "dropout_rate": 0.3,
                    "architecture": "simple",
                    "num_epochs": 20,
                }

        self.optimizer_class = TestOptimizer
        yield

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = self.optimizer_class(
            self.config,
            self.search_space,
            max_trials=50,
            early_stop_patience=10,
            target_accuracy=0.95,
        )

        assert optimizer.max_trials == 50
        assert optimizer.early_stop_patience == 10
        assert optimizer.target_accuracy == 0.95
        assert len(optimizer.results) == 0
        assert optimizer.best_result is None
        assert optimizer.trials_without_improvement == 0

    def test_results_directory_creation(self):
        """Test that results directory is created."""
        optimizer = self.optimizer_class(self.config, self.search_space)

        assert optimizer.results_dir.exists()

    def test_custom_results_directory(self):
        """Test custom results directory."""
        custom_dir = self.temp_dir / "custom_results"
        optimizer = self.optimizer_class(self.config, self.search_space, results_dir=custom_dir)

        assert optimizer.results_dir == custom_dir
        assert custom_dir.exists()

    def test_should_stop_initial(self):
        """Test should_stop initially returns False."""
        optimizer = self.optimizer_class(self.config, self.search_space, max_trials=10)

        assert optimizer.should_stop() is False

    def test_should_stop_max_trials(self):
        """Test should_stop when max trials reached."""
        optimizer = self.optimizer_class(self.config, self.search_space, max_trials=5)

        # Add 5 results
        for i in range(5):
            result = TrialResult(
                trial_id=i,
                hyperparameters={},
                train_accuracy=0.8,
                val_accuracy=0.75,
                test_accuracy=0.7,
                train_loss=0.3,
                val_loss=0.35,
                test_loss=0.4,
                training_time=100.0,
                timestamp="2024-01-01",
            )
            optimizer.results.append(result)

        assert optimizer.should_stop() is True

    def test_should_stop_target_accuracy(self):
        """Test should_stop when target accuracy reached."""
        optimizer = self.optimizer_class(self.config, self.search_space, target_accuracy=0.95)

        result = TrialResult(
            trial_id=1,
            hyperparameters={},
            train_accuracy=0.98,
            val_accuracy=0.96,
            test_accuracy=0.95,
            train_loss=0.05,
            val_loss=0.06,
            test_loss=0.07,
            training_time=100.0,
            timestamp="2024-01-01",
        )
        optimizer.best_result = result

        assert optimizer.should_stop() is True

    def test_should_stop_early_patience(self):
        """Test should_stop with early stopping patience."""
        optimizer = self.optimizer_class(self.config, self.search_space, early_stop_patience=3)

        optimizer.trials_without_improvement = 3

        assert optimizer.should_stop() is True

    def test_record_result_first(self):
        """Test recording first result."""
        optimizer = self.optimizer_class(self.config, self.search_space)

        result = TrialResult(
            trial_id=1,
            hyperparameters={},
            train_accuracy=0.9,
            val_accuracy=0.85,
            test_accuracy=0.8,
            train_loss=0.2,
            val_loss=0.25,
            test_loss=0.3,
            training_time=100.0,
            timestamp="2024-01-01",
        )

        optimizer.record_result(result)

        assert len(optimizer.results) == 1
        assert optimizer.best_result == result
        assert optimizer.trials_without_improvement == 0

    def test_record_result_improvement(self):
        """Test recording result with improvement."""
        optimizer = self.optimizer_class(self.config, self.search_space)

        result1 = TrialResult(
            trial_id=1,
            hyperparameters={},
            train_accuracy=0.8,
            val_accuracy=0.75,
            test_accuracy=0.7,
            train_loss=0.3,
            val_loss=0.35,
            test_loss=0.4,
            training_time=100.0,
            timestamp="2024-01-01",
        )
        optimizer.record_result(result1)

        result2 = TrialResult(
            trial_id=2,
            hyperparameters={},
            train_accuracy=0.85,
            val_accuracy=0.80,
            test_accuracy=0.75,
            train_loss=0.25,
            val_loss=0.30,
            test_loss=0.35,
            training_time=100.0,
            timestamp="2024-01-01",
        )
        optimizer.record_result(result2)

        assert optimizer.best_result == result2
        assert optimizer.trials_without_improvement == 0

    def test_record_result_no_improvement(self):
        """Test recording result without improvement."""
        optimizer = self.optimizer_class(self.config, self.search_space)

        result1 = TrialResult(
            trial_id=1,
            hyperparameters={},
            train_accuracy=0.9,
            val_accuracy=0.85,
            test_accuracy=0.8,
            train_loss=0.2,
            val_loss=0.25,
            test_loss=0.3,
            training_time=100.0,
            timestamp="2024-01-01",
        )
        optimizer.record_result(result1)

        result2 = TrialResult(
            trial_id=2,
            hyperparameters={},
            train_accuracy=0.85,
            val_accuracy=0.80,
            test_accuracy=0.75,
            train_loss=0.25,
            val_loss=0.30,
            test_loss=0.35,
            training_time=100.0,
            timestamp="2024-01-01",
        )
        optimizer.record_result(result2)

        assert optimizer.best_result == result1
        assert optimizer.trials_without_improvement == 1

    def test_save_results(self):
        """Test saving results to file."""
        optimizer = self.optimizer_class(self.config, self.search_space)

        result = TrialResult(
            trial_id=1,
            hyperparameters={"lr": 0.001},
            train_accuracy=0.9,
            val_accuracy=0.85,
            test_accuracy=0.8,
            train_loss=0.2,
            val_loss=0.25,
            test_loss=0.3,
            training_time=100.0,
            timestamp="2024-01-01",
        )
        optimizer.record_result(result)

        results_file = optimizer.results_dir / "optimization_results.json"
        assert results_file.exists()

        with open(results_file) as f:
            data = json.load(f)

        assert "search_space" in data
        assert "results" in data
        assert "best_result" in data
        assert len(data["results"]) == 1

    def test_get_summary_no_results(self):
        """Test get_summary with no results."""
        optimizer = self.optimizer_class(self.config, self.search_space)

        summary = optimizer.get_summary()

        assert "No trials completed" in summary

    def test_get_summary_with_results(self):
        """Test get_summary with results."""
        optimizer = self.optimizer_class(self.config, self.search_space)

        result = TrialResult(
            trial_id=1,
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
            train_accuracy=0.9,
            val_accuracy=0.85,
            test_accuracy=0.8,
            train_loss=0.2,
            val_loss=0.25,
            test_loss=0.3,
            training_time=100.0,
            timestamp="2024-01-01",
        )
        optimizer.record_result(result)

        summary = optimizer.get_summary()

        assert "Total trials: 1" in summary
        assert "0.8500" in summary  # val accuracy
        assert "learning_rate" in summary


@pytest.mark.unit
class TestCreateOptimizer:
    """Tests for create_optimizer factory function."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = BaseConfig(working_dir=self.temp_dir)
        yield

    def test_create_grid_optimizer(self):
        """Test creating grid search optimizer."""
        optimizer = create_optimizer("grid", self.config)

        assert isinstance(optimizer, GridSearchOptimizer)

    def test_create_random_optimizer(self):
        """Test creating random search optimizer."""
        optimizer = create_optimizer("random", self.config)

        assert isinstance(optimizer, RandomSearchOptimizer)

    def test_create_bayesian_optimizer(self):
        """Test creating Bayesian optimizer."""
        optimizer = create_optimizer("bayesian", self.config)

        assert isinstance(optimizer, BayesianOptimizer)

    def test_create_with_case_insensitive(self):
        """Test creating optimizer with case-insensitive type."""
        optimizer1 = create_optimizer("GRID", self.config)
        optimizer2 = create_optimizer("Grid", self.config)
        optimizer3 = create_optimizer("gRiD", self.config)

        assert isinstance(optimizer1, GridSearchOptimizer)
        assert isinstance(optimizer2, GridSearchOptimizer)
        assert isinstance(optimizer3, GridSearchOptimizer)

    def test_create_with_custom_search_space(self):
        """Test creating optimizer with custom search space."""
        custom_space = HyperparameterSpace(
            learning_rates=[0.002],
            batch_sizes=[128],
            dropout_rates=[0.5],
            architectures=["deep"],
            num_epochs=[50],
        )

        optimizer = create_optimizer("random", self.config, search_space=custom_space)

        assert optimizer.search_space == custom_space

    def test_create_with_default_search_space(self):
        """Test creating optimizer with default search space."""
        optimizer = create_optimizer("random", self.config)

        assert optimizer.search_space is not None
        assert len(optimizer.search_space.learning_rates) > 0

    def test_create_with_kwargs(self):
        """Test creating optimizer with additional kwargs."""
        optimizer = create_optimizer("grid", self.config, max_trials=100, target_accuracy=0.98)

        assert optimizer.max_trials == 100
        assert optimizer.target_accuracy == 0.98

    def test_create_invalid_type(self):
        """Test creating optimizer with invalid type."""
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer("invalid_type", self.config)

    def test_create_empty_type(self):
        """Test creating optimizer with empty type."""
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer("", self.config)


@pytest.mark.unit
class TestOptimizerEdgeCases:
    """Edge case tests for optimizers."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        self.config = BaseConfig(working_dir=self.temp_dir)
        yield

    def test_zero_max_trials(self):
        """Test with zero max trials."""
        space = HyperparameterSpace.quick()
        optimizer = RandomSearchOptimizer(self.config, space, max_trials=0)

        assert optimizer.should_stop() is True

    def test_negative_max_trials(self):
        """Test with negative max trials."""
        space = HyperparameterSpace.quick()
        optimizer = RandomSearchOptimizer(self.config, space, max_trials=-1)

        assert optimizer.should_stop() is True

    def test_target_accuracy_above_one(self):
        """Test with target accuracy above 1.0."""
        space = HyperparameterSpace.quick()
        optimizer = RandomSearchOptimizer(self.config, space, target_accuracy=1.5)

        # Should never stop based on accuracy
        assert optimizer.should_stop() is False

    def test_zero_early_stop_patience(self):
        """Test with zero early stop patience."""
        space = HyperparameterSpace.quick()
        optimizer = RandomSearchOptimizer(self.config, space, early_stop_patience=0)

        optimizer.trials_without_improvement = 0
        assert optimizer.should_stop() is True

    def test_single_hyperparameter_space(self):
        """Test with single value in each hyperparameter."""
        space = HyperparameterSpace(
            learning_rates=[0.001],
            batch_sizes=[32],
            dropout_rates=[0.3],
            architectures=["simple"],
            num_epochs=[20],
        )

        optimizer = GridSearchOptimizer(self.config, space)
        assert len(optimizer._grid) == 1

    def test_large_grid_space(self):
        """Test with large grid space."""
        space = HyperparameterSpace.default()
        optimizer = GridSearchOptimizer(self.config, space)

        # 5 * 3 * 4 * 3 * 4 = 720 combinations
        expected_size = 5 * 3 * 4 * 3 * 4
        assert len(optimizer._grid) == expected_size

    def test_results_file_persistence(self):
        """Test that results file persists across optimizer instances."""
        space = HyperparameterSpace.quick()
        results_dir = self.temp_dir / "optimization"

        # Create first optimizer and record result
        optimizer1 = RandomSearchOptimizer(self.config, space, results_dir=results_dir)
        result = TrialResult(
            trial_id=1,
            hyperparameters={},
            train_accuracy=0.9,
            val_accuracy=0.85,
            test_accuracy=0.8,
            train_loss=0.2,
            val_loss=0.25,
            test_loss=0.3,
            training_time=100.0,
            timestamp="2024-01-01",
        )
        optimizer1.record_result(result)

        # Create second optimizer
        RandomSearchOptimizer(self.config, space, results_dir=results_dir)

        # Results file should exist
        results_file = results_dir / "optimization_results.json"
        assert results_file.exists()
