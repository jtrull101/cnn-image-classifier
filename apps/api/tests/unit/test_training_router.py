"""Unit tests for the training router."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from img_classifier_api.app import app
from img_classifier_api.routers import training as training_module
from img_classifier_api.routers.training import JobStatus, TrainingJobRequest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_jobs():
    """Ensure a clean training_jobs store for every test."""
    training_module.training_jobs.clear()
    yield
    training_module.training_jobs.clear()


def _mock_detect_result(dataset_path: str) -> dict:
    """Return a minimal detect() result dict."""
    return {
        "dataset_path": dataset_path,
        "has_train_test_split": False,
        "num_classes": 2,
        "class_names": ["cat", "dog"],
        "total_images": 200,
        "class_distribution": {"cat": 100, "dog": 100},
        "sample_image_shape": (128, 128, 3),
        "recommended_complexity": "medium",
        "is_balanced": True,
    }


# ---------------------------------------------------------------------------
# GET /api/training/dataset-info
# ---------------------------------------------------------------------------


class TestGetDatasetInfo:
    def test_returns_dataset_info(self, client, tmp_path):
        info = _mock_detect_result(str(tmp_path))
        with patch("img_classifier_api.routers.training.DatasetDetector") as mock_cls:
            mock_cls.return_value.detect.return_value = info
            response = client.post(
                "/api/training/dataset-info",
                json={"dataset_path": str(tmp_path)},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["num_classes"] == 2
        assert data["class_names"] == ["cat", "dog"]
        assert data["total_images"] == 200
        assert data["is_balanced"] is True
        assert data["recommended_complexity"] == "medium"

    def test_404_when_path_does_not_exist(self, client, tmp_path):
        missing = str(tmp_path / "no_such_dir")
        response = client.post(
            "/api/training/dataset-info",
            json={"dataset_path": missing},
        )
        assert response.status_code == 404

    def test_400_when_path_is_a_file(self, client, tmp_path):
        file_path = tmp_path / "data.txt"
        file_path.write_text("hello")
        response = client.post(
            "/api/training/dataset-info",
            json={"dataset_path": str(file_path)},
        )
        assert response.status_code == 400

    def test_500_when_detect_raises(self, client, tmp_path):
        with patch("img_classifier_api.routers.training.DatasetDetector") as mock_cls:
            mock_cls.return_value.detect.side_effect = RuntimeError("disk error")
            response = client.post(
                "/api/training/dataset-info",
                json={"dataset_path": str(tmp_path)},
            )
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# POST /api/training/start
# ---------------------------------------------------------------------------


class TestStartTrainingJob:
    def test_creates_queued_job(self, client, tmp_path):
        with patch("img_classifier_api.routers.training.run_training_job"):
            response = client.post(
                "/api/training/start",
                json={"dataset_path": str(tmp_path), "project_name": "proj"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert "job_id" in data
        assert data["message"] != ""

    def test_job_stored_in_memory(self, client, tmp_path):
        with patch("img_classifier_api.routers.training.run_training_job"):
            response = client.post(
                "/api/training/start",
                json={"dataset_path": str(tmp_path), "project_name": "proj"},
            )
        job_id = response.json()["job_id"]
        assert job_id in training_module.training_jobs

    def test_404_when_dataset_missing(self, client, tmp_path):
        response = client.post(
            "/api/training/start",
            json={
                "dataset_path": str(tmp_path / "missing"),
                "project_name": "proj",
            },
        )
        assert response.status_code == 404

    def test_default_parameters_accepted(self, client, tmp_path):
        """Endpoint must accept requests with only required fields."""
        with patch("img_classifier_api.routers.training.run_training_job"):
            response = client.post(
                "/api/training/start",
                json={"dataset_path": str(tmp_path), "project_name": "p"},
            )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/training/status/{job_id}
# ---------------------------------------------------------------------------


class TestGetTrainingStatus:
    def _insert_job(self, job_id: str = "test-job", status: str = "queued") -> dict:
        job = {
            "job_id": job_id,
            "status": status,
            "progress": 0.0,
            "current_epoch": None,
            "total_epochs": 25,
            "current_metrics": None,
            "final_results": None,
            "error": None,
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
        }
        training_module.training_jobs[job_id] = job
        return job

    def test_returns_job_status(self, client):
        self._insert_job("abc", "running")
        response = client.get("/api/training/status/abc")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "abc"
        assert data["status"] == "running"

    def test_404_for_unknown_job(self, client):
        response = client.get("/api/training/status/does-not-exist")
        assert response.status_code == 404

    def test_progress_field_present(self, client):
        self._insert_job("xyz")
        data = client.get("/api/training/status/xyz").json()
        assert "progress" in data


# ---------------------------------------------------------------------------
# GET /api/training/jobs
# ---------------------------------------------------------------------------


class TestListTrainingJobs:
    def _insert_n_jobs(self, n: int) -> list[str]:
        ids = []
        for i in range(n):
            job_id = f"job-{i}"
            training_module.training_jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "current_epoch": None,
                "total_epochs": 10,
                "current_metrics": None,
                "final_results": None,
                "error": None,
                "created_at": datetime.now(),
                "started_at": None,
                "completed_at": None,
            }
            ids.append(job_id)
        return ids

    def test_empty_list(self, client):
        response = client.get("/api/training/jobs")
        assert response.status_code == 200
        data = response.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_returns_all_jobs(self, client):
        self._insert_n_jobs(3)
        data = client.get("/api/training/jobs").json()
        assert data["total"] == 3
        assert len(data["jobs"]) == 3

    def test_limit_parameter(self, client):
        self._insert_n_jobs(10)
        data = client.get("/api/training/jobs?limit=4").json()
        assert len(data["jobs"]) == 4
        assert data["total"] == 10

    def test_skip_parameter(self, client):
        self._insert_n_jobs(5)
        data = client.get("/api/training/jobs?skip=3").json()
        assert len(data["jobs"]) == 2  # 5 - 3 = 2

    def test_skip_beyond_total_returns_empty(self, client):
        self._insert_n_jobs(3)
        data = client.get("/api/training/jobs?skip=10").json()
        assert data["jobs"] == []


# ---------------------------------------------------------------------------
# DELETE /api/training/jobs/{job_id}
# ---------------------------------------------------------------------------


class TestCancelTrainingJob:
    def _insert_job(self, job_id: str = "to-cancel") -> None:
        training_module.training_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "current_epoch": None,
            "total_epochs": 25,
            "current_metrics": None,
            "final_results": None,
            "error": None,
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
        }

    def test_removes_job(self, client):
        self._insert_job("del-me")
        response = client.delete("/api/training/jobs/del-me")
        assert response.status_code == 200
        assert "del-me" not in training_module.training_jobs

    def test_returns_message_response(self, client):
        self._insert_job("bye")
        data = client.delete("/api/training/jobs/bye").json()
        assert "message" in data

    def test_404_for_unknown_job(self, client):
        response = client.delete("/api/training/jobs/ghost")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# run_training_job background task
# ---------------------------------------------------------------------------


class TestRunTrainingJobBackground:
    async def test_marks_job_failed_on_exception(self):
        """run_training_job sets status=failed when the orchestrator raises."""
        job_id = "bg-fail"
        training_module.training_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "current_epoch": None,
            "total_epochs": 5,
            "current_metrics": None,
            "final_results": None,
            "error": None,
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
        }

        request = TrainingJobRequest(
            dataset_path="/nonexistent/path",
            project_name="test",
        )

        # Patch DatasetDetector to raise so we never touch the filesystem
        with patch("img_classifier_api.routers.training.DatasetDetector") as mock_cls:
            mock_cls.return_value.detect.side_effect = RuntimeError("boom")
            await training_module.run_training_job(job_id, request)

        job = training_module.training_jobs[job_id]
        assert job["status"] == JobStatus.FAILED
        assert job["error"] is not None
        assert job["completed_at"] is not None

    async def test_marks_job_completed_on_success(self, tmp_path):
        """run_training_job sets status=completed when orchestrator succeeds."""
        job_id = "bg-ok"
        training_module.training_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "current_epoch": None,
            "total_epochs": 5,
            "current_metrics": None,
            "final_results": None,
            "error": None,
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
        }

        request = TrainingJobRequest(
            dataset_path=str(tmp_path),
            project_name="test",
        )

        fake_results = {
            "metrics": {
                "accuracy": 0.95,
                "loss": 0.1,
                "train_accuracy": 0.94,
                "val_accuracy": 0.93,
                "test_accuracy": 0.92,
            }
        }

        fake_config = MagicMock()
        fake_config.model_path.return_value = tmp_path / "model.keras"

        with (
            patch("img_classifier_api.routers.training.DatasetDetector") as mock_detector,
            patch("img_classifier_api.routers.training.TrainingOrchestrator"),
            patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
        ):
            mock_detector.return_value.create_config.return_value = fake_config
            mock_thread.return_value = fake_results

            await training_module.run_training_job(job_id, request)

        job = training_module.training_jobs[job_id]
        assert job["status"] == JobStatus.COMPLETED
        assert job["progress"] == 100.0
        assert job["final_results"] is not None
        assert job["completed_at"] is not None
