"""Unit tests for the history router."""

import pytest
from collections.abc import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from img_classifier_api.app import app
from img_classifier_api.crud.predictions import create_prediction
from img_classifier_api.database import Base, get_db
from img_classifier_api.models.prediction import PredictionHistory  # noqa: F401 — registers table
from img_classifier_api.routers._constants import _HTMX_REQUEST_HEADER, _HTMX_REQUEST_VALUE

pytestmark = pytest.mark.unit

_HTMX_HEADERS = {_HTMX_REQUEST_HEADER: _HTMX_REQUEST_VALUE}


def _pred(**kwargs) -> dict:
    base = {
        "image_name": "test.jpg",
        "model_name": "model_a",
        "predicted_class": "cat",
        "confidence": 0.85,
        "probabilities": {"cat": 0.85, "dog": 0.15},
    }
    base.update(kwargs)
    return base


@pytest.fixture
def db() -> Generator[Session, None, None]:
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
def client(db: Session) -> Generator[TestClient, None, None]:
    def override() -> Generator[Session, None, None]:
        yield db

    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# GET /api/history — JSON
# ---------------------------------------------------------------------------


class TestGetHistoryJson:
    def test_empty_db_returns_empty_list(self, client: TestClient) -> None:
        data = client.get("/api/history").json()
        assert data["predictions"] == []
        assert data["total"] == 0

    def test_returns_all_predictions(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred())
        create_prediction(db, _pred())
        data = client.get("/api/history").json()
        assert len(data["predictions"]) == 2
        assert data["total"] == 2

    def test_pagination_limit(self, client: TestClient, db: Session) -> None:
        for i in range(5):
            create_prediction(db, _pred(image_name=f"img_{i}.jpg"))
        data = client.get("/api/history?limit=3").json()
        assert len(data["predictions"]) == 3

    def test_pagination_skip(self, client: TestClient, db: Session) -> None:
        for i in range(5):
            create_prediction(db, _pred(image_name=f"img_{i}.jpg"))
        data = client.get("/api/history?skip=4&limit=10").json()
        assert len(data["predictions"]) == 1

    def test_filter_by_model_name(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(model_name="m1"))
        create_prediction(db, _pred(model_name="m2"))
        data = client.get("/api/history?model_name=m1").json()
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["model_name"] == "m1"

    def test_filter_by_user_session(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(user_session="sess1"))
        create_prediction(db, _pred(user_session="sess2"))
        data = client.get("/api/history?user_session=sess1").json()
        assert len(data["predictions"]) == 1

    def test_response_includes_required_fields(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred())
        item = client.get("/api/history").json()["predictions"][0]
        for field in (
            "id",
            "timestamp",
            "image_name",
            "model_name",
            "predicted_class",
            "confidence",
        ):
            assert field in item


# ---------------------------------------------------------------------------
# GET /api/history — HTMX
# ---------------------------------------------------------------------------


class TestGetHistoryHtmx:
    def test_empty_db_returns_html(self, client: TestClient) -> None:
        response = client.get("/api/history", headers=_HTMX_HEADERS)
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_empty_db_shows_no_predictions_message(self, client: TestClient) -> None:
        response = client.get("/api/history", headers=_HTMX_HEADERS)
        assert "No predictions" in response.text

    def test_shows_predicted_class(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(predicted_class="sparrow"))
        assert "sparrow" in client.get("/api/history", headers=_HTMX_HEADERS).text

    def test_shows_image_name(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(image_name="bird.jpg"))
        assert "bird.jpg" in client.get("/api/history", headers=_HTMX_HEADERS).text

    def test_escapes_html_in_image_name(self, client: TestClient, db: Session) -> None:
        """Image filenames with HTML must be escaped in HTMX output (XSS guard)."""
        create_prediction(db, _pred(image_name='<script>alert("xss")</script>.jpg'))
        response = client.get("/api/history", headers=_HTMX_HEADERS)
        assert "<script>" not in response.text
        assert "&lt;script&gt;" in response.text

    def test_escapes_html_in_model_name(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(model_name="<b>evil model</b>"))
        response = client.get("/api/history", headers=_HTMX_HEADERS)
        assert "<b>evil model</b>" not in response.text
        assert "&lt;b&gt;" in response.text

    def test_escapes_html_in_predicted_class(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(predicted_class="<img src=x onerror=alert(1)>"))
        response = client.get("/api/history", headers=_HTMX_HEADERS)
        assert "<img src=x" not in response.text
        assert "&lt;img" in response.text


# ---------------------------------------------------------------------------
# GET /api/history/export
# ---------------------------------------------------------------------------


class TestExportHistory:
    def test_json_export_content_type(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred())
        response = client.get("/api/history/export?export_format=json")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_csv_export_content_type(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred())
        response = client.get("/api/history/export?export_format=csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

    def test_csv_contains_expected_headers(self, client: TestClient) -> None:
        response = client.get("/api/history/export?export_format=csv")
        assert "image_name" in response.text
        assert "model_name" in response.text
        assert "predicted_class" in response.text

    def test_json_export_has_attachment_disposition(self, client: TestClient) -> None:
        response = client.get("/api/history/export?export_format=json")
        assert "attachment" in response.headers.get("content-disposition", "")

    def test_csv_empty_db_contains_only_headers(self, client: TestClient) -> None:
        response = client.get("/api/history/export?export_format=csv")
        lines = [ln for ln in response.text.strip().splitlines() if ln]
        assert len(lines) == 1  # header row only

    def test_start_date_excludes_old_records(self, client: TestClient, db: Session) -> None:
        """Records before start_date must be excluded from the export."""
        import json as _json
        from datetime import datetime, timezone
        from img_classifier_api.models.prediction import PredictionHistory as PH

        old = PH(
            image_name="old.jpg",
            model_name="m",
            predicted_class="cat",
            confidence=0.9,
            probabilities=_json.dumps({"cat": 0.9}),
            timestamp=datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )
        db.add(old)
        db.commit()

        response = client.get("/api/history/export?export_format=json&start_date=2025-01-01")
        assert response.status_code == 200
        records = _json.loads(response.text)
        assert records == []

    def test_end_date_excludes_future_records(self, client: TestClient, db: Session) -> None:
        """Records after end_date must be excluded from the export."""
        import json as _json
        from datetime import datetime, timezone
        from img_classifier_api.models.prediction import PredictionHistory as PH

        recent = PH(
            image_name="recent.jpg",
            model_name="m",
            predicted_class="dog",
            confidence=0.8,
            probabilities=_json.dumps({"dog": 0.8}),
            timestamp=datetime(2030, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
        )
        db.add(recent)
        db.commit()

        response = client.get("/api/history/export?export_format=json&end_date=2025-01-01")
        assert response.status_code == 200
        records = _json.loads(response.text)
        assert records == []


# ---------------------------------------------------------------------------
# GET /api/history/{prediction_id}
# ---------------------------------------------------------------------------


class TestGetHistoryItem:
    def test_returns_correct_item(self, client: TestClient, db: Session) -> None:
        created = create_prediction(db, _pred())
        response = client.get(f"/api/history/{created.id}")
        assert response.status_code == 200
        assert response.json()["id"] == created.id

    def test_404_for_unknown_id(self, client: TestClient) -> None:
        assert client.get("/api/history/999999").status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/history/{prediction_id}
# ---------------------------------------------------------------------------


class TestDeleteHistoryItem:
    def test_deletes_existing_item(self, client: TestClient, db: Session) -> None:
        created = create_prediction(db, _pred())
        response = client.delete(f"/api/history/{created.id}")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_item_absent_after_deletion(self, client: TestClient, db: Session) -> None:
        created = create_prediction(db, _pred())
        client.delete(f"/api/history/{created.id}")
        assert client.get(f"/api/history/{created.id}").status_code == 404

    def test_404_for_unknown_id(self, client: TestClient) -> None:
        assert client.delete("/api/history/999999").status_code == 404


# ---------------------------------------------------------------------------
# POST /api/history/sync
# ---------------------------------------------------------------------------


class TestSyncHistory:
    def _sync_payload(self, **kwargs) -> dict:
        base = {
            "image_name": "img.jpg",
            "model_name": "model_a",
            "predicted_class": "cat",
            "confidence": 0.9,
            "probabilities": {"cat": 0.9},
            "image_hash": "unique_hash_001",
        }
        base.update(kwargs)
        return {"predictions": [base]}

    def test_syncs_new_prediction(self, client: TestClient) -> None:
        data = client.post("/api/history/sync", json=self._sync_payload()).json()
        assert data["synced"] == 1
        assert data["skipped"] == 0
        assert data["total_processed"] == 1

    def test_skips_duplicate_hash(self, client: TestClient, db: Session) -> None:
        create_prediction(db, _pred(image_hash="existing_hash"))
        data = client.post(
            "/api/history/sync",
            json=self._sync_payload(image_hash="existing_hash"),
        ).json()
        assert data["synced"] == 0
        assert data["skipped"] == 1

    def test_empty_list_returns_zero_counts(self, client: TestClient) -> None:
        data = client.post("/api/history/sync", json={"predictions": []}).json()
        assert data["synced"] == 0
        assert data["skipped"] == 0
        assert data["total_processed"] == 0

    def test_predictions_without_hash_always_synced(self, client: TestClient) -> None:
        """Predictions with no image_hash bypass dedup and are always inserted."""
        payload = {
            "predictions": [
                {
                    "image_name": "no_hash.jpg",
                    "model_name": "m",
                    "predicted_class": "x",
                    "confidence": 0.5,
                    "probabilities": {},
                }
            ]
        }
        data = client.post("/api/history/sync", json=payload).json()
        assert data["synced"] == 1

    def test_returns_message_field(self, client: TestClient) -> None:
        data = client.post("/api/history/sync", json={"predictions": []}).json()
        assert "message" in data
