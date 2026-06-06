"""True end-to-end integration tests against a real Docker container.

The container is built and started once per session by fixtures in conftest.py.
All tests share a single container so the DB state accumulates across classes —
that's intentional for TestHistory which verifies record counts.

Run these tests with Docker available:
    uv run pytest apps/api/tests/integration/test_live_api.py -v --timeout=300
"""

import httpx
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.serial]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, anon: httpx.Client) -> None:
        resp = anon.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, anon: httpx.Client) -> None:
        data = anon.get("/health").json()
        assert data["status"] == "healthy"
        assert isinstance(data["models_loaded"], int)
        assert "current_model" in data


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    def test_missing_key_returns_401(self, anon: httpx.Client) -> None:
        resp = anon.get("/api/models")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, wrong_key: httpx.Client) -> None:
        resp = wrong_key.get("/api/models")
        assert resp.status_code == 401

    def test_valid_key_returns_200(self, authed: httpx.Client) -> None:
        resp = authed.get("/api/models")
        assert resp.status_code == 200

    def test_health_no_auth_required(self, anon: httpx.Client) -> None:
        resp = anon.get("/health")
        assert resp.status_code == 200

    def test_home_page_no_auth_required(self, anon: httpx.Client) -> None:
        resp = anon.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_401_includes_www_authenticate_header(self, anon: httpx.Client) -> None:
        resp = anon.get("/api/models")
        assert resp.status_code == 401
        assert "WWW-Authenticate" in resp.headers
        assert resp.headers["WWW-Authenticate"] == "ApiKey"


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


class TestModelManagement:
    def test_list_models_returns_at_least_one_model(self, authed: httpx.Client) -> None:
        data = authed.get("/api/models").json()
        assert len(data["models"]) >= 1

    def test_current_model_is_set(self, authed: httpx.Client) -> None:
        data = authed.get("/api/models").json()
        assert data["current_model"] is not None
        assert len(data["current_model"]) > 0

    def test_get_model_info_by_name(self, authed: httpx.Client) -> None:
        current = authed.get("/api/models").json()["current_model"]
        resp = authed.get(f"/api/models/{current}")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "num_classes" in data
        assert "input_shape" in data

    def test_get_nonexistent_model_returns_404(self, authed: httpx.Client) -> None:
        resp = authed.get("/api/models/does_not_exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class TestPredictions:
    def test_predict_valid_image_returns_200(self, authed: httpx.Client, test_jpeg: bytes) -> None:
        resp = authed.post(
            "/api/predict",
            files={"file": ("test.jpg", test_jpeg, "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self, authed: httpx.Client, test_jpeg: bytes) -> None:
        data = authed.post(
            "/api/predict",
            files={"file": ("test.jpg", test_jpeg, "image/jpeg")},
        ).json()
        assert isinstance(data["class_name"], str)
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["probabilities"], dict)
        assert isinstance(data["model_name"], str)

    def test_predict_probabilities_sum_to_one(self, authed: httpx.Client, test_jpeg: bytes) -> None:
        data = authed.post(
            "/api/predict",
            files={"file": ("test.jpg", test_jpeg, "image/jpeg")},
        ).json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_without_file_returns_422(self, authed: httpx.Client) -> None:
        resp = authed.post("/api/predict")
        assert resp.status_code == 422

    def test_predict_non_image_returns_400(self, authed: httpx.Client) -> None:
        resp = authed.post(
            "/api/predict",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_oversized_file_returns_413(self, authed: httpx.Client) -> None:
        # 11 MB > 10 MB default limit
        big_payload = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)
        resp = authed.post(
            "/api/predict",
            files={"file": ("big.jpg", big_payload, "image/jpeg")},
        )
        assert resp.status_code == 413

    def test_path_traversal_returns_403(self, authed: httpx.Client) -> None:
        resp = authed.post(
            "/api/models/load",
            json={"model_path": "../../etc/passwd"},
        )
        assert resp.status_code == 403

    def test_garbage_bytes_as_image_returns_400(self, authed: httpx.Client) -> None:
        resp = authed.post(
            "/api/predict",
            files={"file": ("garbage.jpg", b"not-an-image", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_empty_content_type_returns_400(self, authed: httpx.Client) -> None:
        resp = authed.post(
            "/api/predict",
            files={"file": ("test.bin", b"\x00" * 100, "")},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestHistory:
    def test_history_endpoint_returns_200(self, authed: httpx.Client) -> None:
        resp = authed.get("/api/history")
        assert resp.status_code == 200

    def test_predict_increments_history(self, authed: httpx.Client, test_jpeg: bytes) -> None:
        # Make a prediction so there is at least one record.
        predict_resp = authed.post(
            "/api/predict",
            files={"file": ("incr.jpg", test_jpeg, "image/jpeg")},
        )
        assert predict_resp.status_code == 200

        history_resp = authed.get("/api/history")
        assert history_resp.status_code == 200
        data = history_resp.json()
        # Response has either a `total` field or a non-empty predictions list.
        assert data.get("total", len(data.get("predictions", []))) >= 1

    def test_history_export_returns_json(self, authed: httpx.Client) -> None:
        resp = authed.get("/api/history/export?export_format=json")
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", "")
