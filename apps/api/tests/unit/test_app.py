"""Comprehensive unit tests for the FastAPI image classification application."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from img_classifier_api.app import ModelManager, app
from img_classifier_api.schemas import PredictionResponse

pytestmark = pytest.mark.unit


@pytest.mark.smoke
class TestSmoke:
    def test_app_starts(self):
        """Ensure the FastAPI application can be instantiated."""
        from fastapi.testclient import TestClient
        from img_classifier_api.app import app

        client = TestClient(app)
        response = client.get("/")

        # Root may render HTML or return a 404 if route not registered, both indicate the app runs
        assert response.status_code in {200, 404}


class TestModelManager:
    """Test the ModelManager class."""

    @pytest.fixture
    def manager(self) -> ModelManager:
        """Create a fresh ModelManager for each test."""
        return ModelManager()

    @pytest.mark.smoke
    def test_initialization(self, manager):
        """Test ModelManager initializes with empty state."""
        assert manager.models == {}
        assert manager.model_info == {}
        assert manager.current_model_name is None
        assert len(manager.default_model_dirs) == 2

    def test_discover_models_no_models(self, manager, tmp_path):
        """Test discover_models when no models exist."""
        # Override default directories with non-existent paths
        manager.default_model_dirs = [tmp_path / "nonexistent"]
        models = manager.discover_models()
        assert models == []

    def test_discover_models_finds_keras_files(self, manager, tmp_path):
        """Test discover_models finds .keras files."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        keras_file = model_dir / "model.keras"
        keras_file.touch()

        manager.default_model_dirs = [model_dir]
        models = manager.discover_models()

        assert len(models) == 1
        assert models[0].name == "model.keras"

    def test_discover_models_finds_h5_files(self, manager, tmp_path):
        """Test discover_models finds .h5 files."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        h5_file = model_dir / "model.h5"
        h5_file.touch()

        manager.default_model_dirs = [model_dir]
        models = manager.discover_models()

        assert len(models) == 1
        assert models[0].name == "model.h5"

    def test_discover_models_finds_both_formats(self, manager, tmp_path):
        """Test discover_models finds both .keras and .h5 files."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model_dir.joinpath("model1.keras").touch()
        model_dir.joinpath("model2.h5").touch()

        manager.default_model_dirs = [model_dir]
        models = manager.discover_models()

        assert len(models) == 2

    def test_load_model_nonexistent_file(self, manager):
        """Test load_model raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            manager.load_model(Path("/nonexistent/model.keras"))

    @patch("tensorflow.keras.models.load_model")
    def test_load_model_success(self, mock_load, manager, tmp_path):
        """Test load_model successfully loads a model."""
        # Create mock model
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 4)
        mock_load.return_value = mock_model

        # Create model file
        model_path = tmp_path / "model.keras"
        model_path.touch()

        # Load model
        name = manager.load_model(model_path, "test_model")

        assert name == "test_model"
        assert "test_model" in manager.models
        assert "test_model" in manager.model_info
        assert manager.current_model_name == "test_model"

    @patch("tensorflow.keras.models.load_model")
    def test_load_model_default_name(self, mock_load, manager, tmp_path):
        """Test load_model uses filename as default name."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 4)
        mock_load.return_value = mock_model

        model_path = tmp_path / "my_model.keras"
        model_path.touch()

        name = manager.load_model(model_path)
        assert name == "my_model"

    @patch("tensorflow.keras.models.load_model")
    def test_load_model_extracts_accuracy_from_filename(self, mock_load, manager, tmp_path):
        """Test load_model extracts accuracy percentage from filename."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 2)
        mock_load.return_value = mock_model

        model_path = tmp_path / "model_95%_final.keras"
        model_path.touch()

        manager.load_model(model_path, "test")
        info = manager.get_info("test")

        assert info.accuracy == 0.95

    @patch("tensorflow.keras.models.load_model")
    def test_load_model_with_metadata_file(self, mock_load, manager, tmp_path):
        """Test load_model reads metadata from .json file."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 3)
        mock_load.return_value = mock_model

        model_path = tmp_path / "model.keras"
        model_path.touch()

        metadata = {"class_names": ["class_a", "class_b", "class_c"], "accuracy": 0.92}
        metadata_path = tmp_path / "model.json"
        metadata_path.write_text(json.dumps(metadata))

        manager.load_model(model_path, "test")
        info = manager.get_info("test")

        assert info.class_names == ["class_a", "class_b", "class_c"]
        assert info.accuracy == 0.92

    @patch("tensorflow.keras.models.load_model")
    def test_load_multiple_models(self, mock_load, manager, tmp_path):
        """Test loading multiple models."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 4)
        mock_load.return_value = mock_model

        model1 = tmp_path / "model1.keras"
        model2 = tmp_path / "model2.keras"
        model1.touch()
        model2.touch()

        manager.load_model(model1, "model1")
        manager.load_model(model2, "model2")

        assert len(manager.models) == 2
        assert manager.current_model_name == "model1"

    @patch("tensorflow.keras.models.load_model")
    def test_get_model_default(self, mock_load, manager, tmp_path):
        """Test get_model uses current model by default."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 4)
        mock_load.return_value = mock_model

        model_path = tmp_path / "model.keras"
        model_path.touch()

        manager.load_model(model_path, "test")
        retrieved = manager.get_model()

        assert retrieved == mock_model

    @patch("tensorflow.keras.models.load_model")
    def test_get_model_by_name(self, mock_load, manager, tmp_path):
        """Test get_model retrieves specific model by name."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 4)
        mock_load.return_value = mock_model

        model_path = tmp_path / "model.keras"
        model_path.touch()

        manager.load_model(model_path, "my_model")
        retrieved = manager.get_model("my_model")

        assert retrieved == mock_model

    def test_get_model_no_model_loaded(self, manager):
        """Test get_model raises error when no model loaded."""
        with pytest.raises(ValueError, match="No model loaded"):
            manager.get_model()

    @patch("tensorflow.keras.models.load_model")
    def test_get_model_nonexistent(self, mock_load, manager, tmp_path):
        """Test get_model raises error for nonexistent model."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 4)
        mock_load.return_value = mock_model

        model_path = tmp_path / "model.keras"
        model_path.touch()

        manager.load_model(model_path, "test")

        with pytest.raises(ValueError, match="Model not found"):
            manager.get_model("nonexistent")

    @patch("tensorflow.keras.models.load_model")
    def test_set_current_model(self, mock_load, manager, tmp_path):
        """Test set_current_model switches active model."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 4)
        mock_load.return_value = mock_model

        model_path = tmp_path / "model.keras"
        model_path.touch()

        manager.load_model(model_path, "model1")
        manager.load_model(model_path, "model2")

        assert manager.current_model_name == "model1"
        manager.set_current_model("model2")
        assert manager.current_model_name == "model2"

    def test_set_current_model_nonexistent(self, manager):
        """Test set_current_model raises error for nonexistent model."""
        with pytest.raises(ValueError, match="Model not found"):
            manager.set_current_model("nonexistent")

    @patch("tensorflow.keras.models.load_model")
    def test_unload_model_removes_from_registry(self, mock_load, manager, tmp_path):
        """Test unload_model removes model from models and model_info dicts."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 4)
        mock_load.return_value = mock_model

        model_path = tmp_path / "model.keras"
        model_path.touch()
        manager.load_model(model_path, "test_model")

        assert "test_model" in manager.models
        manager.unload_model("test_model")

        assert "test_model" not in manager.models
        assert "test_model" not in manager.model_info

    @patch("tensorflow.keras.models.load_model")
    def test_unload_active_model_falls_back_to_another(self, mock_load, manager, tmp_path):
        """Test unloading the active model switches current_model to remaining one."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 4)
        mock_load.return_value = mock_model

        p1 = tmp_path / "m1.keras"
        p2 = tmp_path / "m2.keras"
        p1.touch()
        p2.touch()
        manager.load_model(p1, "m1")
        manager.load_model(p2, "m2")

        assert manager.current_model_name == "m1"
        manager.unload_model("m1")

        assert "m1" not in manager.models
        assert manager.current_model_name == "m2"

    @patch("tensorflow.keras.models.load_model")
    def test_unload_only_model_clears_current(self, mock_load, manager, tmp_path):
        """Test unloading the sole model sets current_model_name to None."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 4)
        mock_load.return_value = mock_model

        model_path = tmp_path / "model.keras"
        model_path.touch()
        manager.load_model(model_path, "only_model")

        assert manager.current_model_name == "only_model"
        manager.unload_model("only_model")

        assert manager.models == {}
        assert manager.current_model_name is None

    @patch("tensorflow.keras.models.load_model")
    def test_unload_non_active_model_keeps_current(self, mock_load, manager, tmp_path):
        """Test unloading a non-active model leaves current_model_name unchanged."""
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 4)
        mock_load.return_value = mock_model

        p1 = tmp_path / "m1.keras"
        p2 = tmp_path / "m2.keras"
        p1.touch()
        p2.touch()
        manager.load_model(p1, "m1")
        manager.load_model(p2, "m2")
        manager.set_current_model("m2")

        manager.unload_model("m1")

        assert "m1" not in manager.models
        assert manager.current_model_name == "m2"

    def test_unload_model_not_found_raises(self, manager):
        """Test unload_model raises ValueError for unknown model name."""
        with pytest.raises(ValueError, match="Model not found"):
            manager.unload_model("ghost")

    @patch("img_classifier_api.app.ModelManager.discover_models")
    @patch("img_classifier_api.app.ModelManager.load_model")
    def test_auto_load_best_model_no_models(self, mock_load, mock_discover, manager):
        """Test auto_load_best_model handles no models found."""
        mock_discover.return_value = []

        manager.auto_load_best_model()

        mock_load.assert_not_called()

    @patch("img_classifier_api.app.ModelManager.discover_models")
    @patch("img_classifier_api.app.ModelManager.load_model")
    def test_auto_load_best_model_finds_highest_accuracy(self, mock_load, mock_discover, manager):
        """Test auto_load_best_model selects highest accuracy model."""
        models = [
            Path("model_80%.keras"),
            Path("model_95%.keras"),
            Path("model_85%.keras"),
        ]
        mock_discover.return_value = models

        manager.auto_load_best_model()

        # Should load the 95% model
        mock_load.assert_called_once()
        call_args = mock_load.call_args[0]
        assert "95%" in str(call_args[0])

    @patch("img_classifier_api.app.ModelManager.discover_models")
    @patch("img_classifier_api.app.ModelManager.load_model")
    def test_auto_load_best_model_fallback_to_first(self, mock_load, mock_discover, manager):
        """Test auto_load_best_model falls back to first model if no accuracy in name."""
        models = [
            Path("model.keras"),
            Path("another_model.keras"),
        ]
        mock_discover.return_value = models

        manager.auto_load_best_model()

        mock_load.assert_called_once()
        call_args = mock_load.call_args[0]
        assert "model.keras" in str(call_args[0])


class TestApiEndpoints:
    """Test FastAPI endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client."""
        return TestClient(app)

    @pytest.mark.smoke
    def test_health_check(self, client):
        """Test /health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "current_model" in data

    def test_home_no_model(self, client):
        """Test / endpoint when no model loaded."""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    @patch("img_classifier_api.app.model_manager.get_info")
    def test_home_with_model(self, mock_info, client):
        """Test / endpoint with model loaded."""
        from img_classifier_api.app import ModelInfo

        mock_info.return_value = ModelInfo(
            name="test_model",
            path="/path/to/model",
            num_classes=4,
            class_names=["a", "b", "c", "d"],
            input_shape=[224, 224, 3],
            accuracy=0.95,
        )

        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_predict_no_file(self, client):
        """Test /predict endpoint without file."""
        response = client.post("/api/predict")

        assert response.status_code == 422  # Validation error

    def test_predict_non_image_file(self, client):
        """Test /predict endpoint with non-image file."""
        data = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/api/predict", files=data)

        assert response.status_code == 400
        assert "must be an image" in response.json()["detail"]

    @patch("img_classifier_api.app.model_manager.get_model")
    @patch("img_classifier_api.app.model_manager.get_info")
    @patch("cv2.imdecode")
    @patch("cv2.resize")
    def test_predict_success(self, mock_resize, mock_decode, mock_info, mock_get_model, client):
        """Test /predict endpoint successfully classifies image."""
        from img_classifier_api.app import ModelInfo

        # Setup mocks
        mock_image = np.random.rand(224, 224, 3).astype(np.uint8)
        mock_decode.return_value = mock_image
        mock_resize.return_value = mock_image

        mock_model = Mock()
        predictions = np.array([[0.1, 0.2, 0.6, 0.1]])  # Class 2 with 0.6 confidence
        mock_model.predict.return_value = predictions
        mock_get_model.return_value = mock_model

        mock_info.return_value = ModelInfo(
            name="test_model",
            path="/path/to/model",
            num_classes=4,
            class_names=["cat", "dog", "bird", "fish"],
            input_shape=[224, 224, 3],
            accuracy=0.95,
        )

        # Create test image
        img_array = np.random.rand(224, 224, 3).astype(np.uint8)
        # Use a simple bytes representation instead of PIL
        img_data = img_array.tobytes()

        response = client.post(
            "/api/predict?save_to_history=false",
            files={"file": ("test.jpg", img_data, "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["class_name"] == "bird"
        assert data["confidence"] == 0.6
        assert data["model_name"] == "test_model"
        assert "probabilities" in data
        assert len(data["probabilities"]) == 4

    @patch("img_classifier_api.app.model_manager.get_model")
    def test_predict_no_model(self, mock_get_model, client):
        """Test /predict when no model loaded."""
        mock_get_model.side_effect = ValueError("No model loaded")

        img_data = b"fake image data"
        response = client.post("/api/predict", files={"file": ("test.jpg", img_data, "image/jpeg")})

        assert response.status_code == 404

    @patch("img_classifier_api.app.model_manager.get_model")
    @patch("img_classifier_api.app.model_manager.get_info")
    @patch("cv2.imdecode")
    def test_predict_invalid_image(self, mock_decode, mock_info, mock_get_model, client):
        """Test /predict with invalid image data."""
        mock_decode.return_value = None

        img_data = b"not valid image data"
        response = client.post("/api/predict", files={"file": ("test.jpg", img_data, "image/jpeg")})

        assert response.status_code == 400
        assert "Unable to decode image" in response.json()["detail"]

    def test_list_models_empty(self, client):
        """Test /api/models endpoint with no models."""
        response = client.get("/api/models")

        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []
        assert data["current_model"] is None

    def test_list_models_with_models(self, client):
        """Test /api/models endpoint with loaded models."""
        from img_classifier_api.app import ModelInfo, model_manager

        model1 = ModelInfo(
            name="model1",
            path="/path/1",
            num_classes=4,
            class_names=["a", "b", "c", "d"],
            input_shape=[224, 224, 3],
        )
        model2 = ModelInfo(
            name="model2",
            path="/path/2",
            num_classes=2,
            class_names=["x", "y"],
            input_shape=[224, 224, 3],
        )

        # Patch model_manager.model_info dictionary directly
        with patch.object(model_manager, "model_info", {"model1": model1, "model2": model2}):
            response = client.get("/api/models")

            assert response.status_code == 200
            data = response.json()
            assert len(data["models"]) == 2

    @patch("img_classifier_api.app.model_manager.get_info")
    def test_get_model_info(self, mock_info_fn, client):
        """Test /api/models/{name} endpoint."""
        from img_classifier_api.app import ModelInfo

        mock_info_fn.return_value = ModelInfo(
            name="test_model",
            path="/path/to/model",
            num_classes=4,
            class_names=["a", "b", "c", "d"],
            input_shape=[224, 224, 3],
            accuracy=0.92,
        )

        response = client.get("/api/models/test_model")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_model"
        assert data["num_classes"] == 4

    @patch("img_classifier_api.app.model_manager.get_info")
    def test_get_model_info_not_found(self, mock_info_fn, client):
        """Test /api/models/{name} for nonexistent model."""
        mock_info_fn.side_effect = ValueError("Model info not found")

        response = client.get("/api/models/nonexistent")

        assert response.status_code == 404

    @patch("img_classifier_api.app.model_manager.set_current_model")
    def test_activate_model(self, mock_set, client):
        """Test /api/models/{name}/activate endpoint."""
        response = client.post("/api/models/test_model/activate")

        assert response.status_code == 200
        data = response.json()
        assert "now active" in data["message"]
        mock_set.assert_called_once_with("test_model")

    @patch("img_classifier_api.app.model_manager.set_current_model")
    def test_activate_model_not_found(self, mock_set, client):
        """Test /api/models/{name}/activate for nonexistent model."""
        mock_set.side_effect = ValueError("Model not found")

        response = client.post("/api/models/nonexistent/activate")

        assert response.status_code == 404

    @patch(
        "img_classifier_api.routers.models._get_allowed_model_dirs",
        return_value=[Path("/").resolve()],
    )
    @patch("img_classifier_api.app.model_manager.load_model")
    def test_load_model_endpoint(self, mock_load, mock_dirs, client):
        """Test /api/models/load endpoint."""
        mock_load.return_value = "loaded_model"

        response = client.post(
            "/api/models/load", json={"model_path": "/path/to/model.keras", "model_name": "test"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "Model loaded successfully" in data["message"]

    @patch(
        "img_classifier_api.routers.models._get_allowed_model_dirs",
        return_value=[Path("/").resolve()],
    )
    @patch("img_classifier_api.app.model_manager.load_model")
    def test_load_model_endpoint_file_not_found(self, mock_load, mock_dirs, client):
        """Test /api/models/load endpoint with nonexistent file."""
        mock_load.side_effect = FileNotFoundError("File not found")

        response = client.post("/api/models/load", json={"model_path": "/nonexistent.keras"})

        assert response.status_code == 404

    @patch(
        "img_classifier_api.routers.models._get_allowed_model_dirs",
        return_value=[Path("/").resolve()],
    )
    @patch("img_classifier_api.app.model_manager.load_model")
    def test_load_model_endpoint_error(self, mock_load, mock_dirs, client):
        """Test /api/models/load endpoint with general error."""
        mock_load.side_effect = Exception("Load failed")

        response = client.post("/api/models/load", json={"model_path": "/path/to/model.keras"})

        assert response.status_code == 500
        assert "Error loading model" in response.json()["detail"]

    @patch("img_classifier_api.app.model_manager.unload_model")
    def test_delete_model_endpoint_success(self, mock_unload, client):
        """Test DELETE /api/models/{name} successfully unloads a model."""
        response = client.delete("/api/models/my_model")

        assert response.status_code == 200
        data = response.json()
        assert "unloaded successfully" in data["message"]
        mock_unload.assert_called_once_with("my_model")

    @patch("img_classifier_api.app.model_manager.unload_model")
    def test_delete_model_endpoint_not_found(self, mock_unload, client):
        """Test DELETE /api/models/{name} returns 404 for unknown model."""
        mock_unload.side_effect = ValueError("Model not found: ghost")

        response = client.delete("/api/models/ghost")

        assert response.status_code == 404
        assert "Model not found" in response.json()["detail"]

    def test_delete_model_endpoint_path_not_intercepted_by_load(self, client):
        """Ensure DELETE /api/models/load is not mistakenly routed to the load POST endpoint."""
        # This verifies route ordering: POST /models/load should not intercept DELETE requests.
        # If routing is wrong, this would 405 or 422 rather than reaching the DELETE handler.
        with patch("img_classifier_api.app.model_manager.unload_model") as mock_unload:
            response = client.delete("/api/models/load")
            # "load" is treated as a model name for DELETE; should reach unload handler
            assert response.status_code in {200, 404}
            # Either it tries to unload "load" (200 if mock doesn't raise) or 404
            mock_unload.assert_called_once_with("load")


class TestUploadSizeLimit:
    """Edge-case tests for the upload size guard."""

    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    @patch("img_classifier_api.routers.predictions._MAX_UPLOAD_BYTES", 100)
    def test_file_over_limit_returns_413(self, client):
        """A file one byte over the limit must be rejected with HTTP 413."""
        data = b"x" * 101
        response = client.post("/api/predict", files={"file": ("test.jpg", data, "image/jpeg")})
        assert response.status_code == 413

    @patch("img_classifier_api.routers.predictions._MAX_UPLOAD_BYTES", 100)
    def test_file_at_exact_limit_passes_size_check(self, client):
        """A file exactly at the limit must not be rejected with 413.

        The request will still fail (no model loaded → 404 or 400),
        but the size check itself must succeed.
        """
        data = b"x" * 100
        response = client.post("/api/predict", files={"file": ("test.jpg", data, "image/jpeg")})
        assert response.status_code != 413


class TestBatchPredictionBoundaries:
    """Edge-case tests for the batch prediction endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    @patch("img_classifier_api.routers.predictions._run_prediction")
    def test_single_file_batch_returns_one_result(self, mock_run, client):
        """Batch with exactly 1 file should produce exactly 1 result entry."""
        from img_classifier_api.routers.predictions import PredictionResponse as PredResp

        mock_run.return_value = (
            PredResp(
                class_name="cat",
                confidence=0.9,
                probabilities={"cat": 0.9, "dog": 0.1},
                model_name="m",
            ),
            b"bytes",
            Mock(),
        )
        response = client.post(
            "/api/predict/batch",
            files=[("files", ("img.jpg", b"data", "image/jpeg"))],
            params={"save_to_history": "false"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["successful"] == 1


class TestPredictionResponse:
    """Test the PredictionResponse model."""

    @pytest.mark.smoke
    def test_prediction_response_creation(self):
        """Test creating a PredictionResponse."""
        response = PredictionResponse(
            class_name="cat",
            confidence=0.95,
            probabilities={"cat": 0.95, "dog": 0.05},
            model_name="my_model",
        )

        assert response.class_name == "cat"
        assert response.confidence == 0.95
        assert response.model_name == "my_model"
        assert len(response.probabilities) == 2

    def test_prediction_response_json_serializable(self):
        """Test PredictionResponse can be serialized to JSON."""
        response = PredictionResponse(
            class_name="bird",
            confidence=0.87,
            probabilities={"bird": 0.87, "plane": 0.13},
            model_name="classifier_v2",
        )

        json_str = response.model_dump_json()
        assert "bird" in json_str
        assert "0.87" in json_str
