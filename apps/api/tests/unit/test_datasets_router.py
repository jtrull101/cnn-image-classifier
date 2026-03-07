"""Unit tests for the datasets router."""

import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from img_classifier_api.app import app
from img_classifier_api.routers import datasets as datasets_module
from img_classifier_api.routers.datasets import (
    DownloadUrlRequest,
    KaggleDownloadRequest,
    _dataset_name_from_url,
    _safe_dataset_path,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_download_jobs():
    """Ensure a clean download_jobs store for every test."""
    datasets_module.download_jobs.clear()
    yield
    datasets_module.download_jobs.clear()


@pytest.fixture
def datasets_dir(tmp_path):
    """Patch _datasets_dir to use a temp directory."""
    with patch("img_classifier_api.routers.datasets._datasets_dir", return_value=tmp_path):
        yield tmp_path


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestDatasetNameFromUrl:
    def test_strips_zip_extension(self):
        assert _dataset_name_from_url("https://example.com/data.zip", None) == "data"

    def test_strips_tar_gz_extension(self):
        assert _dataset_name_from_url("https://example.com/data.tar.gz", None) == "data"

    def test_strips_tgz_extension(self):
        assert _dataset_name_from_url("https://example.com/data.tgz", None) == "data"

    def test_no_extension_returns_filename(self):
        assert _dataset_name_from_url("https://example.com/mydata", None) == "mydata"

    def test_override_takes_priority(self):
        assert _dataset_name_from_url("https://example.com/data.zip", "custom") == "custom"

    def test_trailing_slash_stripped(self):
        assert _dataset_name_from_url("https://example.com/data.zip/", None) == "data"


class TestSafeDatasetPath:
    def test_valid_name(self, tmp_path):
        result = _safe_dataset_path(tmp_path, "my_dataset")
        assert result == (tmp_path / "my_dataset").resolve()

    def test_path_traversal_raises(self, tmp_path):
        with pytest.raises(Exception):
            _safe_dataset_path(tmp_path, "../secret")


# ---------------------------------------------------------------------------
# GET /api/datasets
# ---------------------------------------------------------------------------


class TestListDatasets:
    def test_empty_when_no_datasets_dir(self, client, tmp_path):
        missing = tmp_path / "no_datasets"
        with patch("img_classifier_api.routers.datasets._datasets_dir", return_value=missing):
            response = client.get("/api/datasets")
        assert response.status_code == 200
        data = response.json()
        assert data["datasets"] == []
        assert data["total"] == 0

    def test_lists_subdirectories(self, client, tmp_path):
        (tmp_path / "cats").mkdir()
        (tmp_path / "dogs").mkdir()
        (tmp_path / "cats" / "img1.jpg").touch()
        with patch("img_classifier_api.routers.datasets._datasets_dir", return_value=tmp_path):
            response = client.get("/api/datasets")
        data = response.json()
        assert data["total"] == 2
        names = [d["name"] for d in data["datasets"]]
        assert "cats" in names
        assert "dogs" in names

    def test_does_not_list_files(self, client, tmp_path):
        (tmp_path / "file.txt").touch()
        (tmp_path / "actual_dataset").mkdir()
        with patch("img_classifier_api.routers.datasets._datasets_dir", return_value=tmp_path):
            response = client.get("/api/datasets")
        data = response.json()
        assert data["total"] == 1
        assert data["datasets"][0]["name"] == "actual_dataset"

    def test_num_items_counts_contents(self, client, tmp_path):
        ds = tmp_path / "myds"
        ds.mkdir()
        (ds / "a").touch()
        (ds / "b").touch()
        with patch("img_classifier_api.routers.datasets._datasets_dir", return_value=tmp_path):
            response = client.get("/api/datasets")
        assert response.json()["datasets"][0]["num_items"] == 2

    def test_datasets_dir_path_in_response(self, client, tmp_path):
        with patch("img_classifier_api.routers.datasets._datasets_dir", return_value=tmp_path):
            response = client.get("/api/datasets")
        assert response.json()["datasets_dir"] == str(tmp_path)


# ---------------------------------------------------------------------------
# POST /api/datasets/download/url
# ---------------------------------------------------------------------------


class TestDownloadDatasetFromUrl:
    def test_queues_job(self, client):
        with patch("img_classifier_api.routers.datasets._run_url_download"):
            response = client.post(
                "/api/datasets/download/url",
                json={"url": "https://example.com/data.zip"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert "job_id" in data
        assert data["message"] != ""

    def test_job_stored_in_memory(self, client):
        with patch("img_classifier_api.routers.datasets._run_url_download"):
            response = client.post(
                "/api/datasets/download/url",
                json={"url": "https://example.com/data.zip"},
            )
        job_id = response.json()["job_id"]
        assert job_id in datasets_module.download_jobs

    def test_default_extract_true(self, client):
        with patch("img_classifier_api.routers.datasets._run_url_download"):
            response = client.post(
                "/api/datasets/download/url",
                json={"url": "https://example.com/data.zip"},
            )
        assert response.status_code == 200

    def test_accepts_authorization_field(self, client):
        with patch("img_classifier_api.routers.datasets._run_url_download"):
            response = client.post(
                "/api/datasets/download/url",
                json={
                    "url": "https://example.com/data.zip",
                    "authorization": "Bearer mytoken",
                },
            )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/datasets/download/kaggle
# ---------------------------------------------------------------------------


class TestDownloadDatasetFromKaggle:
    def test_queues_job(self, client):
        with patch("img_classifier_api.routers.datasets._run_kaggle_download"):
            response = client.post(
                "/api/datasets/download/kaggle",
                json={
                    "dataset": "owner/my-dataset",
                    "kaggle_username": "user",
                    "kaggle_key": "abc123",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert "job_id" in data

    def test_400_when_dataset_missing_slash(self, client):
        response = client.post(
            "/api/datasets/download/kaggle",
            json={
                "dataset": "nodash",
                "kaggle_username": "user",
                "kaggle_key": "abc123",
            },
        )
        assert response.status_code == 400

    def test_job_stored_in_memory(self, client):
        with patch("img_classifier_api.routers.datasets._run_kaggle_download"):
            response = client.post(
                "/api/datasets/download/kaggle",
                json={
                    "dataset": "owner/dataset",
                    "kaggle_username": "u",
                    "kaggle_key": "k",
                },
            )
        job_id = response.json()["job_id"]
        assert job_id in datasets_module.download_jobs


# ---------------------------------------------------------------------------
# GET /api/datasets/download/{job_id}
# ---------------------------------------------------------------------------


class TestGetDownloadStatus:
    def _insert_job(self, job_id: str = "test-job", status: str = "queued") -> dict:
        job = {
            "job_id": job_id,
            "status": status,
            "message": "Download queued",
            "dataset_name": None,
            "dataset_path": None,
            "error": None,
            "created_at": datetime.now(),
            "completed_at": None,
        }
        datasets_module.download_jobs[job_id] = job
        return job

    def test_returns_job_status(self, client):
        self._insert_job("abc", "running")
        response = client.get("/api/datasets/download/abc")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "abc"
        assert data["status"] == "running"

    def test_404_for_unknown_job(self, client):
        response = client.get("/api/datasets/download/does-not-exist")
        assert response.status_code == 404

    def test_all_fields_present(self, client):
        self._insert_job("xyz", "completed")
        data = client.get("/api/datasets/download/xyz").json()
        for field in ("job_id", "status", "message", "dataset_name", "dataset_path", "error"):
            assert field in data


# ---------------------------------------------------------------------------
# DELETE /api/datasets/{dataset_name}
# ---------------------------------------------------------------------------


class TestDeleteDataset:
    def test_deletes_existing_dataset(self, client, tmp_path):
        ds = tmp_path / "cats"
        ds.mkdir()
        with patch("img_classifier_api.routers.datasets._datasets_dir", return_value=tmp_path):
            response = client.delete("/api/datasets/cats")
        assert response.status_code == 200
        assert not ds.exists()

    def test_returns_message_response(self, client, tmp_path):
        (tmp_path / "dogs").mkdir()
        with patch("img_classifier_api.routers.datasets._datasets_dir", return_value=tmp_path):
            data = client.delete("/api/datasets/dogs").json()
        assert "message" in data

    def test_404_when_dataset_missing(self, client, tmp_path):
        with patch("img_classifier_api.routers.datasets._datasets_dir", return_value=tmp_path):
            response = client.delete("/api/datasets/nonexistent")
        assert response.status_code == 404

    def test_400_when_not_a_directory(self, client, tmp_path):
        (tmp_path / "file.txt").touch()
        with patch("img_classifier_api.routers.datasets._datasets_dir", return_value=tmp_path):
            response = client.delete("/api/datasets/file.txt")
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Background task: _run_url_download
# ---------------------------------------------------------------------------


class TestRunUrlDownload:
    async def test_marks_completed_on_success(self, tmp_path):
        job_id = "url-ok"
        datasets_module.download_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "queued",
            "dataset_name": None,
            "dataset_path": None,
            "error": None,
            "created_at": datetime.now(),
            "completed_at": None,
        }

        # Create a real zip file in a separate dir (not in datasets_dir) to avoid same-file error
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        zip_src = src_dir / "data.zip"
        with zipfile.ZipFile(zip_src, "w") as zf:
            zf.writestr("img1.jpg", b"fake")

        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        request = DownloadUrlRequest(url="https://example.com/data.zip")

        def fake_download(url: str, dest: Path, headers: dict) -> None:
            import shutil

            shutil.copy(zip_src, dest)

        with (
            patch("img_classifier_api.routers.datasets._datasets_dir", return_value=datasets_dir),
            patch("img_classifier_api.routers.datasets._download_file", side_effect=fake_download),
        ):
            await datasets_module._run_url_download(job_id, request)

        job = datasets_module.download_jobs[job_id]
        assert job["status"] == "completed"
        assert job["dataset_name"] == "data"
        assert job["completed_at"] is not None

    async def test_marks_failed_on_exception(self, tmp_path):
        job_id = "url-fail"
        datasets_module.download_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "queued",
            "dataset_name": None,
            "dataset_path": None,
            "error": None,
            "created_at": datetime.now(),
            "completed_at": None,
        }

        request = DownloadUrlRequest(url="https://example.com/data.zip")

        with (
            patch("img_classifier_api.routers.datasets._datasets_dir", return_value=tmp_path),
            patch(
                "img_classifier_api.routers.datasets._download_file",
                side_effect=RuntimeError("network error"),
            ),
        ):
            await datasets_module._run_url_download(job_id, request)

        job = datasets_module.download_jobs[job_id]
        assert job["status"] == "failed"
        assert "network error" in job["error"]
        assert job["completed_at"] is not None

    async def test_skips_extraction_when_extract_false(self, tmp_path):
        job_id = "url-noextract"
        datasets_module.download_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "queued",
            "dataset_name": None,
            "dataset_path": None,
            "error": None,
            "created_at": datetime.now(),
            "completed_at": None,
        }

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        zip_src = src_dir / "data.zip"
        with zipfile.ZipFile(zip_src, "w") as zf:
            zf.writestr("img1.jpg", b"fake")

        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        request = DownloadUrlRequest(url="https://example.com/data.zip", extract=False)

        def fake_download(url: str, dest: Path, headers: dict) -> None:
            import shutil

            shutil.copy(zip_src, dest)

        with (
            patch("img_classifier_api.routers.datasets._datasets_dir", return_value=datasets_dir),
            patch("img_classifier_api.routers.datasets._download_file", side_effect=fake_download),
        ):
            await datasets_module._run_url_download(job_id, request)

        job = datasets_module.download_jobs[job_id]
        assert job["status"] == "completed"
        # File should remain as-is, not extracted
        assert job["dataset_path"] is not None
        assert job["dataset_path"].endswith("data.zip")


# ---------------------------------------------------------------------------
# Background task: _run_kaggle_download
# ---------------------------------------------------------------------------


class TestRunKaggleDownload:
    async def test_marks_completed_on_success(self, tmp_path):
        job_id = "kaggle-ok"
        datasets_module.download_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "queued",
            "dataset_name": None,
            "dataset_path": None,
            "error": None,
            "created_at": datetime.now(),
            "completed_at": None,
        }

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        zip_src = src_dir / "alzheimers.zip"
        with zipfile.ZipFile(zip_src, "w") as zf:
            zf.writestr("img1.jpg", b"fake")

        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        request = KaggleDownloadRequest(
            dataset="neuroimaging/alzheimers",
            kaggle_username="user",
            kaggle_key="key123",
        )

        def fake_download(url: str, dest: Path, headers: dict) -> None:
            import shutil

            assert "Basic" in headers.get("Authorization", "")
            shutil.copy(zip_src, dest)

        with (
            patch("img_classifier_api.routers.datasets._datasets_dir", return_value=datasets_dir),
            patch("img_classifier_api.routers.datasets._download_file", side_effect=fake_download),
        ):
            await datasets_module._run_kaggle_download(job_id, request)

        job = datasets_module.download_jobs[job_id]
        assert job["status"] == "completed"
        assert job["dataset_name"] == "alzheimers"
        assert job["completed_at"] is not None

    async def test_marks_failed_on_exception(self, tmp_path):
        job_id = "kaggle-fail"
        datasets_module.download_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "queued",
            "dataset_name": None,
            "dataset_path": None,
            "error": None,
            "created_at": datetime.now(),
            "completed_at": None,
        }

        request = KaggleDownloadRequest(
            dataset="owner/dataset",
            kaggle_username="user",
            kaggle_key="key",
        )

        with (
            patch("img_classifier_api.routers.datasets._datasets_dir", return_value=tmp_path),
            patch(
                "img_classifier_api.routers.datasets._download_file",
                side_effect=RuntimeError("auth failed"),
            ),
        ):
            await datasets_module._run_kaggle_download(job_id, request)

        job = datasets_module.download_jobs[job_id]
        assert job["status"] == "failed"
        assert "auth failed" in job["error"]

    async def test_uses_basic_auth_header(self, tmp_path):
        """Kaggle download must send a Basic auth header derived from username:key."""
        import base64

        job_id = "kaggle-auth"
        datasets_module.download_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "queued",
            "dataset_name": None,
            "dataset_path": None,
            "error": None,
            "created_at": datetime.now(),
            "completed_at": None,
        }

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        zip_src = src_dir / "ds.zip"
        with zipfile.ZipFile(zip_src, "w") as zf:
            zf.writestr("f.txt", b"x")

        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        captured_headers: dict = {}

        def fake_download(url: str, dest: Path, headers: dict) -> None:
            import shutil

            captured_headers.update(headers)
            shutil.copy(zip_src, dest)

        request = KaggleDownloadRequest(
            dataset="owner/ds",
            kaggle_username="alice",
            kaggle_key="secret",
        )

        with (
            patch("img_classifier_api.routers.datasets._datasets_dir", return_value=datasets_dir),
            patch("img_classifier_api.routers.datasets._download_file", side_effect=fake_download),
        ):
            await datasets_module._run_kaggle_download(job_id, request)

        auth = captured_headers.get("Authorization", "")
        assert auth.startswith("Basic ")
        decoded = base64.b64decode(auth[len("Basic ") :]).decode()
        assert decoded == "alice:secret"
