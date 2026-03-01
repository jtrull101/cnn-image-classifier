"""Security-focused tests for the FastAPI image classification application.

Covers:
- API key authentication (item 1)
- Path traversal prevention in model loading (item 2)
- File upload size limits (item 4)
- Unauthenticated endpoint access rejection
"""

import io
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 4, height: int = 4) -> bytes:
    """Return a minimal valid PNG file as bytes."""
    import struct
    import zlib

    def chunk(name: bytes, data: bytes) -> bytes:
        c = name + data
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(c))

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    # Build minimal IDAT (raw, unfiltered RGB rows)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def _large_bytes(size_mb: int = 11) -> bytes:
    """Return a byte string larger than the default upload limit."""
    return b"X" * (size_mb * 1024 * 1024)


# ---------------------------------------------------------------------------
# API key authentication tests
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    """API key middleware correctly enforces authentication when configured."""

    @pytest.fixture(autouse=True)
    def _set_api_key(self, monkeypatch):
        """Enable authentication by setting the env var for every test."""
        monkeypatch.setenv("IMG_CLASSIFIER_API_KEY", "test-secret-key")

    @pytest.fixture
    def client(self):
        # Import inside fixture so env var is already set
        from img_classifier_api.app import app

        return TestClient(app, raise_server_exceptions=False)

    def test_predict_without_key_returns_401(self, client):
        png = _make_png_bytes()
        response = client.post(
            "/api/predict",
            files={"file": ("test.png", png, "image/png")},
        )
        assert response.status_code == 401

    def test_predict_with_wrong_key_returns_401(self, client):
        png = _make_png_bytes()
        response = client.post(
            "/api/predict",
            files={"file": ("test.png", png, "image/png")},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_list_models_without_key_returns_401(self, client):
        response = client.get("/api/models")
        assert response.status_code == 401

    def test_history_without_key_returns_401(self, client):
        response = client.get("/api/history")
        assert response.status_code == 401

    def test_training_start_without_key_returns_401(self, client):
        response = client.post(
            "/api/training/start",
            json={"dataset_path": "/tmp/fake", "project_name": "test"},
        )
        assert response.status_code == 401

    def test_health_endpoint_does_not_require_key(self, client):
        """Health check must remain public."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_home_page_does_not_require_key(self, client):
        """Web UI must remain accessible without a key."""
        response = client.get("/")
        assert response.status_code == 200

    def test_valid_key_passes_through(self, client):
        """A correct key must not be rejected (endpoint may still fail for other reasons)."""
        png = _make_png_bytes()
        response = client.post(
            "/api/predict",
            files={"file": ("test.png", png, "image/png")},
            headers={"X-API-Key": "test-secret-key"},
        )
        # Auth passed — endpoint can return 404/422/500 for other reasons (no model loaded)
        assert response.status_code != 401


class TestApiKeyAuthDisabled:
    """When no API key is configured, all routes are open."""

    @pytest.fixture(autouse=True)
    def _clear_api_key(self, monkeypatch):
        monkeypatch.delenv("IMG_CLASSIFIER_API_KEY", raising=False)

    @pytest.fixture
    def client(self):
        from img_classifier_api.app import app

        return TestClient(app, raise_server_exceptions=False)

    def test_models_endpoint_open_without_key(self, client):
        response = client.get("/api/models")
        # Should not be 401 — may be 200 or 404 depending on loaded models
        assert response.status_code != 401


# ---------------------------------------------------------------------------
# Path traversal prevention tests
# ---------------------------------------------------------------------------


class TestPathTraversal:
    """Model loading endpoint rejects paths outside allowed directories."""

    @pytest.fixture(autouse=True)
    def _clear_api_key(self, monkeypatch):
        monkeypatch.delenv("IMG_CLASSIFIER_API_KEY", raising=False)

    @pytest.fixture
    def client(self):
        from img_classifier_api.app import app

        return TestClient(app, raise_server_exceptions=False)

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "../../etc/passwd",
            "/etc/shadow",
            "../../../../etc/hosts",
            "/root/.ssh/id_rsa",
            "/tmp/../etc/passwd",
        ],
    )
    def test_path_traversal_returns_403(self, client, malicious_path):
        response = client.post(
            "/api/models/load",
            json={"model_path": malicious_path},
        )
        assert response.status_code == 403, (
            f"Expected 403 for path '{malicious_path}', got {response.status_code}"
        )

    def test_allowed_path_does_not_return_403(self, client, tmp_path):
        """A path inside an allowed dir returns 404 (not found) not 403 (forbidden)."""
        # Temporarily make tmp_path one of the allowed dirs via env var
        with patch.dict(os.environ, {"IMG_CLASSIFIER_MODEL_DIR": str(tmp_path)}):
            response = client.post(
                "/api/models/load",
                json={"model_path": str(tmp_path / "model.keras")},
            )
        # 404 means the file was not there — traversal check passed
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# File upload size limit tests
# ---------------------------------------------------------------------------


class TestFileUploadSizeLimit:
    """Oversized uploads are rejected before inference."""

    @pytest.fixture(autouse=True)
    def _clear_api_key(self, monkeypatch):
        monkeypatch.delenv("IMG_CLASSIFIER_API_KEY", raising=False)
        # Set a tiny limit so tests run fast — patch the cached module-level constant directly
        monkeypatch.setattr("img_classifier_api.routers.predictions._MAX_UPLOAD_BYTES", 1024)

    @pytest.fixture
    def client(self):
        from img_classifier_api.app import app

        return TestClient(app, raise_server_exceptions=False)

    def test_oversized_file_returns_413(self, client):
        big_data = b"X" * (2 * 1024)  # 2 KB — above 1 KB limit
        response = client.post(
            "/api/predict",
            files={"file": ("big.png", io.BytesIO(big_data), "image/png")},
        )
        assert response.status_code == 413

    def test_valid_size_is_accepted(self, client):
        """A small valid-sized PNG must pass the size gate (may fail for other reasons)."""
        png = _make_png_bytes()  # Tiny PNG well under 1 KB
        response = client.post(
            "/api/predict",
            files={"file": ("small.png", io.BytesIO(png), "image/png")},
        )
        # 413 must not occur — other errors (no model, decode error) are acceptable
        assert response.status_code != 413

    def test_non_image_content_type_rejected(self, client):
        response = client.post(
            "/api/predict",
            files={"file": ("payload.txt", io.BytesIO(b"malicious"), "text/plain")},
        )
        assert response.status_code == 400
