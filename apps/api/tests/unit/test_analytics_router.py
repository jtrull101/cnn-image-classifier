"""Unit tests for the analytics router."""

from collections.abc import Generator
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from img_classifier_api.app import app
from img_classifier_api.crud.predictions import create_prediction
from img_classifier_api.database import Base, get_db
from img_classifier_api.models.prediction import PredictionHistory
from img_classifier_api.routers._constants import _HTMX_REQUEST_HEADER, _HTMX_REQUEST_VALUE


pytestmark = pytest.mark.unit

_HTMX_HEADERS = {_HTMX_REQUEST_HEADER: _HTMX_REQUEST_VALUE}


def _pred(**kwargs) -> dict:
    base = {
        "image_name": "test.jpg",
        "model_name": "model_a",
        "predicted_class": "cat",
        "confidence": 0.9,
        "probabilities": {"cat": 0.9, "dog": 0.1},
    }
    base.update(kwargs)
    return base


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


# ---------------------------------------------------------------------------
# GET /api/analytics/summary — JSON
# ---------------------------------------------------------------------------


class TestAnalyticsSummaryJson:
    def test_empty_db_returns_zero_totals(self, client: TestClient) -> None:
        data = client.get("/api/analytics/summary").json()
        assert data["total_predictions"] == 0
        assert data["predictions_by_model"] == {}
        assert data["predictions_by_class"] == {}
        assert data["average_confidence"] == 0.0

    def test_counts_predictions_by_model(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(model_name="m1"))
        create_prediction(db, _pred(model_name="m1"))
        create_prediction(db, _pred(model_name="m2"))
        data = client.get("/api/analytics/summary").json()
        assert data["predictions_by_model"]["m1"] == 2
        assert data["predictions_by_model"]["m2"] == 1

    def test_counts_predictions_by_class(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(predicted_class="cat"))
        create_prediction(db, _pred(predicted_class="dog"))
        create_prediction(db, _pred(predicted_class="cat"))
        data = client.get("/api/analytics/summary").json()
        assert data["predictions_by_class"]["cat"] == 2
        assert data["predictions_by_class"]["dog"] == 1

    def test_calculates_average_confidence(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(confidence=0.8))
        create_prediction(db, _pred(confidence=0.6))
        data = client.get("/api/analytics/summary").json()
        assert data["average_confidence"] == pytest.approx(0.7)

    def test_average_confidence_by_model(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(model_name="m1", confidence=0.9))
        create_prediction(db, _pred(model_name="m2", confidence=0.5))
        data = client.get("/api/analytics/summary").json()
        assert data["average_confidence_by_model"]["m1"] == pytest.approx(0.9)
        assert data["average_confidence_by_model"]["m2"] == pytest.approx(0.5)

    def test_status_200(self, client: TestClient) -> None:
        assert client.get("/api/analytics/summary").status_code == 200


# ---------------------------------------------------------------------------
# GET /api/analytics/summary — HTMX
# ---------------------------------------------------------------------------


class TestAnalyticsSummaryHtmx:
    def test_returns_html_content_type(self, client: TestClient) -> None:
        response = client.get("/api/analytics/summary", headers=_HTMX_HEADERS)
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_empty_db_shows_no_data_placeholder(self, client: TestClient) -> None:
        response = client.get("/api/analytics/summary", headers=_HTMX_HEADERS)
        assert "No data" in response.text

    def test_shows_model_name(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(model_name="resnet50"))
        response = client.get("/api/analytics/summary", headers=_HTMX_HEADERS)
        assert "resnet50" in response.text

    def test_shows_predicted_class(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(predicted_class="sparrow"))
        response = client.get("/api/analytics/summary", headers=_HTMX_HEADERS)
        assert "sparrow" in response.text

    def test_escapes_html_in_model_name(self, client: TestClient, db: Session) -> None:
        """Model names with HTML must be escaped in HTMX output (XSS guard)."""
        create_prediction(db, _pred(model_name='<script>alert("xss")</script>'))
        response = client.get("/api/analytics/summary", headers=_HTMX_HEADERS)
        assert "<script>" not in response.text
        assert "&lt;script&gt;" in response.text

    def test_escapes_html_in_predicted_class(self, client: TestClient, db: Session) -> None:
        """Class names with HTML must be escaped in HTMX output (XSS guard)."""
        create_prediction(db, _pred(predicted_class="<b>evil</b>"))
        response = client.get("/api/analytics/summary", headers=_HTMX_HEADERS)
        assert "<b>evil</b>" not in response.text
        assert "&lt;b&gt;" in response.text


# ---------------------------------------------------------------------------
# GET /api/analytics/performance
# ---------------------------------------------------------------------------


class TestAnalyticsPerformance:
    def test_empty_db_returns_empty_models_dict(self, client: TestClient) -> None:
        data = client.get("/api/analytics/performance").json()
        assert data["models"] == {}

    def test_returns_per_model_totals(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(model_name="m1"))
        create_prediction(db, _pred(model_name="m1"))
        data = client.get("/api/analytics/performance").json()
        assert data["models"]["m1"]["total_predictions"] == 2

    def test_returns_confidence_stats(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(model_name="m1", confidence=0.9))
        create_prediction(db, _pred(model_name="m1", confidence=0.7))
        m1 = client.get("/api/analytics/performance").json()["models"]["m1"]
        assert m1["average_confidence"] == pytest.approx(0.8)
        assert m1["min_confidence"] == pytest.approx(0.7)
        assert m1["max_confidence"] == pytest.approx(0.9)

    def test_class_distribution_per_model(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(model_name="m1", predicted_class="cat"))
        create_prediction(db, _pred(model_name="m1", predicted_class="dog"))
        create_prediction(db, _pred(model_name="m1", predicted_class="cat"))
        dist = client.get("/api/analytics/performance").json()["models"]["m1"]["class_distribution"]
        assert dist["cat"] == 2
        assert dist["dog"] == 1

    def test_multiple_models_are_independent(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(model_name="m1"))
        create_prediction(db, _pred(model_name="m2"))
        models = client.get("/api/analytics/performance").json()["models"]
        assert "m1" in models
        assert "m2" in models
        assert models["m1"]["total_predictions"] == 1
        assert models["m2"]["total_predictions"] == 1

    def test_start_date_excludes_old_records(self, client: TestClient, db: Session) -> None:
        """Records before start_date must be absent from performance results."""
        import json as _json

        from img_classifier_api.models.prediction import PredictionHistory as PH

        old = PH(
            image_name="old.jpg",
            model_name="old_model",
            predicted_class="cat",
            confidence=0.9,
            probabilities=_json.dumps({"cat": 0.9}),
            timestamp=datetime(2020, 1, 1),
        )
        db.add(old)
        db.commit()

        data = client.get("/api/analytics/performance?start_date=2025-01-01").json()
        assert "old_model" not in data["models"]

    def test_end_date_excludes_future_records(self, client: TestClient, db: Session) -> None:
        """Records after end_date must be absent from performance results."""
        import json as _json

        from img_classifier_api.models.prediction import PredictionHistory as PH

        future = PH(
            image_name="future.jpg",
            model_name="future_model",
            predicted_class="dog",
            confidence=0.8,
            probabilities=_json.dumps({"dog": 0.8}),
            timestamp=datetime(2030, 6, 1),
        )
        db.add(future)
        db.commit()

        data = client.get("/api/analytics/performance?end_date=2025-01-01").json()
        assert "future_model" not in data["models"]
