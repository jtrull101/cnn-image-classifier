"""Unit tests for the predictions router — history saving and compare endpoints."""

from collections.abc import Generator
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from img_classifier_api.app import app
from img_classifier_api.crud.predictions import get_predictions
from img_classifier_api.database import Base, get_db
from img_classifier_api.models.prediction import PredictionHistory
from img_classifier_api.schemas import ModelInfo


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> Generator[Session]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    _Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = _Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def client(db: Session) -> Generator[TestClient]:
    def override() -> Generator[Session]:
        yield db

    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_db, None)


def _fake_model_info(name: str = "test_model") -> ModelInfo:
    return ModelInfo(
        name=name,
        path="/path/model.keras",
        num_classes=2,
        class_names=["cat", "dog"],
        input_shape=[64, 64, 3],
    )


# ---------------------------------------------------------------------------
# POST /api/predict  — save_to_history=True
# ---------------------------------------------------------------------------


class TestPredictSavesToHistory:
    """Verify the history-saving branch in predict_image."""

    @patch("img_classifier_api.app.model_manager.get_model")
    @patch("img_classifier_api.app.model_manager.get_info")
    @patch("cv2.imencode")
    @patch("cv2.resize")
    @patch("cv2.imdecode")
    def test_creates_history_record(
        self,
        mock_decode: Mock,
        mock_resize: Mock,
        mock_imencode: Mock,
        mock_info: Mock,
        mock_get_model: Mock,
        client: TestClient,
        db: Session,
    ) -> None:
        """A successful prediction with save_to_history=True must persist one record."""
        fake_image = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_decode.return_value = fake_image
        mock_resize.return_value = fake_image
        mock_imencode.return_value = (True, np.zeros(10, dtype=np.uint8))

        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.2, 0.8]])
        mock_get_model.return_value = mock_model
        mock_info.return_value = _fake_model_info()

        assert get_predictions(db) == []

        response = client.post(
            "/api/predict",
            files={"file": ("test.jpg", b"fake-img-data", "image/jpeg")},
        )
        assert response.status_code == 200
        records = get_predictions(db)
        assert len(records) == 1
        assert records[0].model_name == "test_model"
        assert records[0].predicted_class == "dog"

    @patch("img_classifier_api.app.model_manager.get_model")
    @patch("img_classifier_api.app.model_manager.get_info")
    @patch("cv2.imencode")
    @patch("cv2.resize")
    @patch("cv2.imdecode")
    def test_record_stores_image_hash(
        self,
        mock_decode: Mock,
        mock_resize: Mock,
        mock_imencode: Mock,
        mock_info: Mock,
        mock_get_model: Mock,
        client: TestClient,
        db: Session,
    ) -> None:
        """Saved record must have a non-None image_hash (SHA-256 of upload bytes)."""
        fake_image = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_decode.return_value = fake_image
        mock_resize.return_value = fake_image
        mock_imencode.return_value = (True, np.zeros(10, dtype=np.uint8))
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.9, 0.1]])
        mock_get_model.return_value = mock_model
        mock_info.return_value = _fake_model_info()

        client.post(
            "/api/predict",
            files={"file": ("img.jpg", b"some-bytes", "image/jpeg")},
        )
        record = get_predictions(db)[0]
        assert record.image_hash is not None
        assert len(record.image_hash) == 64  # SHA-256 hex digest

    @patch("img_classifier_api.app.model_manager.get_model")
    @patch("img_classifier_api.app.model_manager.get_info")
    @patch("cv2.imencode")
    @patch("cv2.resize")
    @patch("cv2.imdecode")
    def test_record_stores_thumbnail(
        self,
        mock_decode: Mock,
        mock_resize: Mock,
        mock_imencode: Mock,
        mock_info: Mock,
        mock_get_model: Mock,
        client: TestClient,
        db: Session,
    ) -> None:
        """Saved record must have a base64 image_thumbnail."""
        fake_image = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_decode.return_value = fake_image
        mock_resize.return_value = fake_image
        mock_imencode.return_value = (True, np.array([0xFF, 0xD8], dtype=np.uint8))
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.7, 0.3]])
        mock_get_model.return_value = mock_model
        mock_info.return_value = _fake_model_info()

        client.post(
            "/api/predict",
            files={"file": ("img.jpg", b"data", "image/jpeg")},
        )
        record = get_predictions(db)[0]
        assert record.image_thumbnail is not None
        assert len(record.image_thumbnail) > 0

    @patch("img_classifier_api.app.model_manager.get_model")
    @patch("img_classifier_api.app.model_manager.get_info")
    @patch("cv2.imencode")
    @patch("cv2.resize")
    @patch("cv2.imdecode")
    def test_no_record_when_save_false(
        self,
        mock_decode: Mock,
        mock_resize: Mock,
        mock_imencode: Mock,
        mock_info: Mock,
        mock_get_model: Mock,
        client: TestClient,
        db: Session,
    ) -> None:
        """save_to_history=false must not persist any record."""
        fake_image = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_decode.return_value = fake_image
        mock_resize.return_value = fake_image
        mock_imencode.return_value = (True, np.zeros(10, dtype=np.uint8))
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.5, 0.5]])
        mock_get_model.return_value = mock_model
        mock_info.return_value = _fake_model_info()

        response = client.post(
            "/api/predict?save_to_history=false",
            files={"file": ("img.jpg", b"data", "image/jpeg")},
        )
        assert response.status_code == 200
        assert get_predictions(db) == []


# ---------------------------------------------------------------------------
# POST /api/predict/compare
# ---------------------------------------------------------------------------


class TestComparePredict:
    """Tests for the multi-model comparison endpoint."""

    @pytest.fixture
    def plain_client(self) -> TestClient:
        return TestClient(app)

    @patch("img_classifier_api.routers.predictions._run_prediction")
    def test_returns_predictions_for_requested_models(
        self, mock_run: AsyncMock, plain_client: TestClient
    ) -> None:
        from img_classifier_api.schemas import PredictionResponse

        mock_run.return_value = (
            PredictionResponse(
                class_name="cat",
                confidence=0.9,
                probabilities={"cat": 0.9, "dog": 0.1},
                model_name="model_a",
            ),
            b"bytes",
            Mock(),
        )

        response = plain_client.post(
            "/api/predict/compare?model_names=model_a",
            files={"file": ("img.jpg", b"data", "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "model_a" in data["predictions"]
        assert data["predictions"]["model_a"]["class_name"] == "cat"

    @patch("img_classifier_api.routers.predictions._run_prediction")
    def test_error_in_one_model_returns_error_entry(
        self, mock_run: AsyncMock, plain_client: TestClient
    ) -> None:
        """When a model raises during comparison, the entry has class_name='ERROR'."""
        mock_run.side_effect = Exception("inference failed")

        response = plain_client.post(
            "/api/predict/compare?model_names=bad_model",
            files={"file": ("img.jpg", b"data", "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"]["bad_model"]["class_name"] == "ERROR"
        assert data["predictions"]["bad_model"]["confidence"] == 0.0

    def test_no_models_available_returns_400(self, plain_client: TestClient) -> None:
        """Returns 400 when no models are loaded and no model_names specified."""
        with patch.object(app.state.model_manager, "models", {}):
            response = plain_client.post(
                "/api/predict/compare",
                files={"file": ("img.jpg", b"data", "image/jpeg")},
            )
        assert response.status_code == 400

    @patch("img_classifier_api.routers.predictions._run_prediction")
    def test_returns_filename_in_response(
        self, mock_run: AsyncMock, plain_client: TestClient
    ) -> None:
        from img_classifier_api.schemas import PredictionResponse

        mock_run.return_value = (
            PredictionResponse(
                class_name="dog",
                confidence=0.8,
                probabilities={"dog": 0.8, "cat": 0.2},
                model_name="m",
            ),
            b"b",
            Mock(),
        )
        response = plain_client.post(
            "/api/predict/compare?model_names=m",
            files={"file": ("photo.jpg", b"data", "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["image_name"] == "photo.jpg"
