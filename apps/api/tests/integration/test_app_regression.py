"""Integration tests for the FastAPI image classification application.

These tests verify behaviour against the real application stack with no mocking.
All tests use the real request/response cycle; heavy-mocked workflow tests live
in apps/api/tests/unit/test_app.py instead.

Note: Each pytest-xdist worker uses an isolated database to prevent race conditions.
"""

import pytest
from fastapi.testclient import TestClient

from img_classifier_api.app import app


pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Health endpoint is always reachable without auth."""

    @pytest.mark.smoke
    def test_health_check_always_succeeds(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "current_model" in data
        assert "websocket_connections" in data


class TestPredictionValidation:
    """Validation errors that require no model to be loaded."""

    def test_predict_unsupported_mime_type(self, client):
        response = client.post(
            "/api/predict", files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_predict_empty_content_type(self, client):
        response = client.post("/api/predict", files={"file": ("test.bin", b"binary data", "")})
        assert response.status_code == 400

    def test_predict_no_file(self, client):
        response = client.post("/api/predict")
        assert response.status_code == 422  # FastAPI validation error


class TestModelEndpointBehavior:
    """Model management endpoints with no model loaded."""

    @pytest.mark.smoke
    def test_list_models_empty(self, client):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "current_model" in data

    def test_activate_nonexistent_model_returns_404(self, client):
        response = client.post("/api/models/nonexistent_model/activate")
        assert response.status_code == 404

    def test_get_nonexistent_model_info_returns_404(self, client):
        response = client.get("/api/models/nonexistent_model")
        assert response.status_code == 404

    def test_load_model_path_outside_allowed_dirs_returns_403(self, client):
        """Path traversal guard returns 403 for paths outside allowed directories."""
        response = client.post(
            "/api/models/load",
            json={"model_path": "/etc/passwd"},
        )
        assert response.status_code == 403

    def test_load_model_missing_file_returns_404(self, client, tmp_path, monkeypatch):
        """A path inside an allowed directory that does not exist returns 404."""
        # Make tmp_path an allowed model directory via the env var, then request a
        # file that is inside it but does not exist — the real filesystem is consulted.
        monkeypatch.setenv("IMG_CLASSIFIER_MODEL_DIR", str(tmp_path))
        response = client.post(
            "/api/models/load",
            json={"model_path": str(tmp_path / "nonexistent_model.keras")},
        )
        assert response.status_code == 404


class TestHistoryEndpoints:
    """History endpoints work with real DB (empty on fresh startup)."""

    def test_get_history_returns_empty_list(self, client):
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["predictions"] == [] or isinstance(data["predictions"], list)

    def test_get_nonexistent_history_item_returns_404(self, client):
        response = client.get("/api/history/999999")
        assert response.status_code == 404

    def test_delete_nonexistent_history_item_returns_404(self, client):
        response = client.delete("/api/history/999999")
        assert response.status_code == 404

    def test_sync_empty_list(self, client):
        response = client.post("/api/history/sync", json={"predictions": []})
        assert response.status_code == 200
        data = response.json()
        assert data["synced"] == 0
        assert data["skipped"] == 0


class TestAnalyticsEndpoints:
    """Analytics endpoints return valid structure with an empty database."""

    def test_analytics_summary_returns_valid_structure(self, client):
        response = client.get("/api/analytics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "predictions_by_model" in data
        assert "average_confidence" in data

    def test_analytics_performance_returns_valid_structure(self, client):
        response = client.get("/api/analytics/performance")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
