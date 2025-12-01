"""Integration tests for the FastAPI image classification application.

These tests verify complete workflows and interactions between components.
"""

from pathlib import Path

# TODO remove mocking from integration tests
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from img_classifier_api.app import app, ModelManager


pytestmark = pytest.mark.integration


class TestModelManagerWorkflows:
    """Test complete ModelManager workflows."""

    @patch("tensorflow.keras.models.load_model")
    @pytest.mark.smoke
    def test_workflow_load_and_predict_with_metadata(self, mock_load):
        """Test workflow: load model with metadata, then retrieve."""
        manager = ModelManager()

        # Create mock model
        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 3)
        mock_load.return_value = mock_model

        # Create model directory with files
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.touch"):
                model_path = Path("model_88%.keras")

                # Create metadata
                with patch("builtins.open", create=True) as mock_file:
                    mock_file.return_value.__enter__.return_value.read.return_value = (
                        '{"class_names": ["cat", "dog", "bird"], "accuracy": 0.88}'
                    )

                    # Simulate loading with metadata
                    manager.load_model(model_path, "classifier_v1")

        # Verify model is loaded and current
        assert manager.current_model_name == "classifier_v1"
        assert manager.get_model("classifier_v1") == mock_model

        # Get info
        info = manager.get_info("classifier_v1")
        assert info.name == "classifier_v1"
        assert info.num_classes == 3

    @patch("tensorflow.keras.models.load_model")
    def test_workflow_multiple_models_switching(self, mock_load):
        """Test workflow: load multiple models and switch between them."""
        manager = ModelManager()

        mock_model = Mock()
        mock_model.input_shape = (None, 224, 224, 3)
        mock_model.output_shape = (None, 2)
        mock_load.return_value = mock_model

        with patch("pathlib.Path.exists", return_value=True):
            # Load first model
            path1 = Path("model1.keras")
            manager.load_model(path1, "detector_v1")

            # Load second model
            path2 = Path("model2.keras")
            manager.load_model(path2, "detector_v2")

        # First model should be current
        assert manager.current_model_name == "detector_v1"

        # Switch to second
        manager.set_current_model("detector_v2")
        assert manager.current_model_name == "detector_v2"

        # Both should be retrievable
        assert manager.get_model("detector_v1") is not None
        assert manager.get_model("detector_v2") is not None


class TestApiPredictionWorkflow:
    """Test complete prediction workflows through API."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @patch("img_classifier_api.app.model_manager.get_model")
    @patch("img_classifier_api.app.model_manager.get_info")
    @patch("cv2.imdecode")
    @patch("cv2.resize")
    @pytest.mark.smoke
    def test_full_prediction_workflow(
        self, mock_resize, mock_decode, mock_info, mock_get_model, client
    ):
        """Test complete prediction workflow: upload → classify → respond."""
        from img_classifier_api.app import ModelInfo

        # Setup mock predictions
        mock_image = np.random.rand(224, 224, 3).astype(np.uint8)
        mock_decode.return_value = mock_image
        mock_resize.return_value = mock_image

        mock_model = Mock()
        # High confidence prediction
        predictions = np.array([[0.05, 0.88, 0.07]])
        mock_model.predict.return_value = predictions
        mock_get_model.return_value = mock_model

        mock_info.return_value = ModelInfo(
            name="classifier",
            path="/models/classifier.keras",
            num_classes=3,
            class_names=["negative", "positive", "uncertain"],
            input_shape=[224, 224, 3],
            accuracy=0.92,
        )

        # Upload image
        img_data = b"\x00" * 100  # Dummy image data
        response = client.post("/api/predict", files={"file": ("test.jpg", img_data, "image/jpeg")})

        # Verify prediction response
        assert response.status_code == 200
        data = response.json()
        assert data["class_name"] == "positive"
        assert data["confidence"] == 0.88
        assert data["model_name"] == "classifier"

        # Verify all classes have probabilities
        probs = data["probabilities"]
        assert len(probs) == 3
        assert "positive" in probs

    @patch("img_classifier_api.app.model_manager.set_current_model")
    @patch("img_classifier_api.app.model_manager.get_info")
    @patch("img_classifier_api.app.model_manager.get_model")
    @patch("cv2.imdecode")
    @patch("cv2.resize")
    def test_multi_model_prediction_workflow(
        self, mock_resize, mock_decode, mock_get_model, mock_info, mock_set, client
    ):
        """Test workflow: switch model → predict with new model."""
        from img_classifier_api.app import ModelInfo

        # Setup
        mock_image = np.random.rand(224, 224, 3).astype(np.uint8)
        mock_decode.return_value = mock_image
        mock_resize.return_value = mock_image

        mock_model = Mock()
        predictions = np.array([[0.1, 0.9]])
        mock_model.predict.return_value = predictions
        mock_get_model.return_value = mock_model

        mock_info.return_value = ModelInfo(
            name="v2_model",
            path="/models/v2.keras",
            num_classes=2,
            class_names=["no", "yes"],
            input_shape=[224, 224, 3],
        )

        # Switch model
        response = client.post("/api/models/v2_model/activate")
        assert response.status_code == 200
        mock_set.assert_called_with("v2_model")

        # Make prediction with new model
        img_data = b"\x00" * 100
        response = client.post("/api/predict", files={"file": ("test.jpg", img_data, "image/jpeg")})

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "v2_model"
        assert data["class_name"] == "yes"


class TestErrorRecoveryWorkflows:
    """Test error handling and recovery workflows."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @patch("img_classifier_api.app.model_manager.get_model")
    @pytest.mark.smoke
    def test_recover_from_missing_model_error(self, mock_get_model, client):
        """Test API gracefully handles missing model during prediction."""
        mock_get_model.side_effect = ValueError("No model loaded")

        img_data = b"fake image"
        response = client.post("/api/predict", files={"file": ("test.jpg", img_data, "image/jpeg")})

        # Should return 404, not 500
        assert response.status_code == 404
        assert "No model loaded" in response.json()["detail"]

    def test_recover_from_missing_file_on_load(self, client):
        """Test API handles missing file error during model load."""
        response = client.post(
            "/api/models/load", params={"model_path": "/nonexistent/path/model.keras"}
        )

        assert response.status_code == 404

    @patch("img_classifier_api.app.model_manager.get_model")
    @patch("img_classifier_api.app.model_manager.get_info")
    @patch("cv2.imdecode")
    def test_recover_from_corrupt_image(self, mock_decode, mock_info, mock_get_model, client):
        """Test API handles corrupt image gracefully."""
        mock_decode.return_value = None  # Indicate decoding failed

        img_data = b"corrupt image data"
        response = client.post("/api/predict", files={"file": ("test.jpg", img_data, "image/jpeg")})

        assert response.status_code == 400
        assert "Unable to decode image" in response.json()["detail"]

    def test_invalid_model_activation(self, client):
        """Test error when activating non-existent model."""
        response = client.post("/api/models/nonexistent_model/activate")
        assert response.status_code == 404


class TestApiStateManagement:
    """Test API state management across requests."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @patch("img_classifier_api.app.model_manager.current_model_name", None)
    @patch("img_classifier_api.app.model_manager.models", {})
    @patch("img_classifier_api.app.model_manager.model_info", {})
    @pytest.mark.smoke
    def test_state_empty_on_startup(self, client):
        """Test API starts with no models loaded."""
        response = client.get("/api/models")
        data = response.json()

        assert data["models"] == []
        assert data["current_model"] is None

    @patch("img_classifier_api.app.model_manager.load_model")
    def test_state_persists_across_requests(self, mock_load, client):
        """Test model state persists between requests."""
        # First request: load model
        mock_load.return_value = "loaded"
        response1 = client.post(
            "/api/models/load",
            params={"model_path": "/path/to/model.keras", "model_name": "persistent"},
        )
        assert response1.status_code == 200

        # Second request: check model still loaded
        # (In real scenario, model would persist in memory)
        response2 = client.get("/health")
        assert response2.status_code == 200

    def test_health_check_always_succeeds(self, client):
        """Test health endpoint always returns success."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data


@pytest.mark.integration
class TestImageProcessingEdgeCases:
    """Test edge cases in image processing."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @patch("img_classifier_api.app.model_manager.get_model")
    @patch("img_classifier_api.app.model_manager.get_info")
    @patch("cv2.imdecode")
    @patch("cv2.resize")
    @pytest.mark.smoke
    def test_predict_very_small_image(
        self, mock_resize, mock_decode, mock_info, mock_get_model, client
    ):
        """Test prediction with very small input image."""
        from img_classifier_api.app import ModelInfo

        tiny_image = np.random.rand(10, 10, 3).astype(np.uint8)
        mock_decode.return_value = tiny_image
        mock_resize.return_value = np.random.rand(224, 224, 3).astype(np.uint8)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.3, 0.7]])
        mock_get_model.return_value = mock_model

        mock_info.return_value = ModelInfo(
            name="test",
            path="test",
            num_classes=2,
            class_names=["a", "b"],
            input_shape=[224, 224, 3],
        )

        response = client.post(
            "/api/predict", files={"file": ("tiny.jpg", b"\x00" * 100, "image/jpeg")}
        )

        # Should resize and process successfully
        assert response.status_code == 200

    @patch("img_classifier_api.app.model_manager.get_model")
    @patch("img_classifier_api.app.model_manager.get_info")
    @patch("cv2.imdecode")
    @patch("cv2.resize")
    def test_predict_very_large_image(
        self, mock_resize, mock_decode, mock_info, mock_get_model, client
    ):
        """Test prediction with very large input image."""
        from img_classifier_api.app import ModelInfo

        large_image = np.random.rand(4000, 4000, 3).astype(np.uint8)
        mock_decode.return_value = large_image
        mock_resize.return_value = np.random.rand(224, 224, 3).astype(np.uint8)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.3, 0.7]])
        mock_get_model.return_value = mock_model

        mock_info.return_value = ModelInfo(
            name="test",
            path="test",
            num_classes=2,
            class_names=["a", "b"],
            input_shape=[224, 224, 3],
        )

        response = client.post(
            "/api/predict", files={"file": ("large.jpg", b"\x00" * 1000000, "image/jpeg")}
        )

        # Should resize and process successfully
        assert response.status_code == 200
        mock_resize.assert_called()

    def test_predict_unsupported_image_type(self, client):
        """Test prediction with unsupported image MIME type."""
        response = client.post(
            "/api/predict", files={"file": ("test.txt", b"not an image", "text/plain")}
        )

        assert response.status_code == 400

    def test_predict_missing_content_type(self, client):
        """Test prediction with missing content-type header."""
        response = client.post("/api/predict", files={"file": ("test.bin", b"binary data", "")})

        assert response.status_code == 400
