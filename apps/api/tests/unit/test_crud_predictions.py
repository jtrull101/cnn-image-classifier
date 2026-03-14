"""Unit tests for prediction CRUD operations using an in-memory SQLite database."""

import json
from collections.abc import Generator
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from img_classifier_api.crud.predictions import (
    count_predictions,
    create_prediction,
    delete_prediction,
    export_predictions,
    get_analytics_summary,
    get_prediction_by_id,
    get_predictions,
)
from img_classifier_api.schemas import ExportFormat
from img_classifier_api.database import Base
from img_classifier_api.models.prediction import PredictionHistory  # noqa: F401 — registers table with Base


pytestmark = pytest.mark.unit


@pytest.fixture
def db() -> Generator[Session, None, None]:
    """In-memory SQLite session — no file I/O, safe for parallel tests."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    _Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = _Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def _pred(**kwargs) -> dict:
    """Return a minimal valid prediction data dict, optionally overriding fields."""
    base = {
        "image_name": "test.jpg",
        "model_name": "test_model",
        "predicted_class": "cat",
        "confidence": 0.85,
        "probabilities": {"cat": 0.85, "dog": 0.15},
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# create_prediction
# ---------------------------------------------------------------------------


class TestCreatePrediction:
    def test_record_is_persisted_and_returned(self, db):
        result = create_prediction(db, _pred())
        assert result.id is not None
        assert result.image_name == "test.jpg"
        assert result.model_name == "test_model"
        assert result.predicted_class == "cat"
        assert result.confidence == pytest.approx(0.85)

    def test_probabilities_are_stored_as_json_string(self, db):
        result = create_prediction(db, _pred(probabilities={"cat": 0.9, "dog": 0.1}))
        stored = json.loads(result.probabilities)
        assert stored == {"cat": 0.9, "dog": 0.1}

    def test_optional_fields_default_to_none(self, db):
        result = create_prediction(db, _pred())
        assert result.image_hash is None
        assert result.image_thumbnail is None
        assert result.user_session is None

    def test_optional_fields_stored_when_provided(self, db):
        result = create_prediction(
            db, _pred(image_hash="abc123", image_thumbnail="b64data", user_session="sess-xyz")
        )
        assert result.image_hash == "abc123"
        assert result.image_thumbnail == "b64data"
        assert result.user_session == "sess-xyz"

    def test_multiple_creates_get_distinct_ids(self, db):
        r1 = create_prediction(db, _pred())
        r2 = create_prediction(db, _pred())
        assert r1.id != r2.id


# ---------------------------------------------------------------------------
# get_predictions
# ---------------------------------------------------------------------------


class TestGetPredictions:
    def test_empty_db_returns_empty_list(self, db):
        assert get_predictions(db) == []

    def test_returns_all_records_by_default(self, db):
        create_prediction(db, _pred())
        create_prediction(db, _pred())
        assert len(get_predictions(db)) == 2

    def test_skip_beyond_total_returns_empty_list(self, db):
        create_prediction(db, _pred())
        assert get_predictions(db, skip=100) == []

    def test_limit_restricts_number_of_results(self, db):
        for _ in range(5):
            create_prediction(db, _pred())
        assert len(get_predictions(db, limit=3)) == 3

    def test_skip_and_limit_paginate_correctly(self, db):
        for i in range(5):
            create_prediction(db, _pred(image_name=f"img_{i}.jpg"))
        first = get_predictions(db, skip=0, limit=3)
        second = get_predictions(db, skip=3, limit=3)
        assert len(first) == 3
        assert len(second) == 2

    def test_filter_by_model_name(self, db):
        create_prediction(db, _pred(model_name="model_a"))
        create_prediction(db, _pred(model_name="model_b"))
        results = get_predictions(db, model_name="model_a")
        assert len(results) == 1
        assert results[0].model_name == "model_a"

    def test_filter_model_name_none_returns_all(self, db):
        create_prediction(db, _pred(model_name="model_a"))
        create_prediction(db, _pred(model_name="model_b"))
        assert len(get_predictions(db, model_name=None)) == 2

    def test_filter_by_user_session(self, db):
        create_prediction(db, _pred(user_session="sess_a"))
        create_prediction(db, _pred(user_session="sess_b"))
        results = get_predictions(db, user_session="sess_a")
        assert len(results) == 1
        assert results[0].user_session == "sess_a"

    def test_filter_user_session_none_returns_all(self, db):
        create_prediction(db, _pred(user_session="sess_a"))
        create_prediction(db, _pred(user_session="sess_b"))
        assert len(get_predictions(db, user_session=None)) == 2

    def test_combined_model_and_session_filter(self, db):
        create_prediction(db, _pred(model_name="m1", user_session="s1"))
        create_prediction(db, _pred(model_name="m1", user_session="s2"))
        create_prediction(db, _pred(model_name="m2", user_session="s1"))
        results = get_predictions(db, model_name="m1", user_session="s1")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# get_prediction_by_id
# ---------------------------------------------------------------------------


class TestGetPredictionById:
    def test_returns_correct_record(self, db):
        created = create_prediction(db, _pred())
        result = get_prediction_by_id(db, created.id)
        assert result is not None
        assert result.id == created.id

    def test_returns_none_when_not_found(self, db):
        assert get_prediction_by_id(db, 9999) is None


# ---------------------------------------------------------------------------
# delete_prediction
# ---------------------------------------------------------------------------


class TestDeletePrediction:
    def test_returns_true_when_deleted(self, db):
        created = create_prediction(db, _pred())
        assert delete_prediction(db, created.id) is True

    def test_record_absent_after_deletion(self, db):
        created = create_prediction(db, _pred())
        delete_prediction(db, created.id)
        assert get_prediction_by_id(db, created.id) is None

    def test_returns_false_when_not_found(self, db):
        assert delete_prediction(db, 9999) is False


# ---------------------------------------------------------------------------
# count_predictions
# ---------------------------------------------------------------------------


class TestCountPredictions:
    def test_empty_db_returns_zero(self, db):
        assert count_predictions(db) == 0

    def test_counts_all_records(self, db):
        create_prediction(db, _pred())
        create_prediction(db, _pred())
        assert count_predictions(db) == 2

    def test_filter_by_model_name(self, db):
        create_prediction(db, _pred(model_name="model_a"))
        create_prediction(db, _pred(model_name="model_b"))
        assert count_predictions(db, model_name="model_a") == 1

    def test_filter_model_name_none_counts_all(self, db):
        create_prediction(db, _pred(model_name="model_a"))
        create_prediction(db, _pred(model_name="model_b"))
        assert count_predictions(db, model_name=None) == 2

    def test_filter_by_user_session(self, db):
        create_prediction(db, _pred(user_session="sess_a"))
        create_prediction(db, _pred(user_session="sess_b"))
        assert count_predictions(db, user_session="sess_a") == 1

    def test_filter_user_session_none_counts_all(self, db):
        create_prediction(db, _pred(user_session="sess_a"))
        create_prediction(db, _pred(user_session="sess_b"))
        assert count_predictions(db, user_session=None) == 2

    def test_combined_model_and_session_filter(self, db):
        create_prediction(db, _pred(model_name="m1", user_session="s1"))
        create_prediction(db, _pred(model_name="m1", user_session="s2"))
        create_prediction(db, _pred(model_name="m2", user_session="s1"))
        assert count_predictions(db, model_name="m1", user_session="s1") == 1

    def test_unmatched_filter_returns_zero(self, db):
        create_prediction(db, _pred(model_name="model_a"))
        assert count_predictions(db, model_name="nonexistent") == 0


# ---------------------------------------------------------------------------
# export_predictions
# ---------------------------------------------------------------------------


class TestExportPredictions:
    def test_json_export_is_valid_list(self, db):
        create_prediction(db, _pred())
        parsed = json.loads(export_predictions(db, export_format=ExportFormat.JSON))
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["image_name"] == "test.jpg"

    def test_json_export_empty_db_returns_empty_array(self, db):
        parsed = json.loads(export_predictions(db, export_format=ExportFormat.JSON))
        assert parsed == []

    def test_csv_export_contains_headers_and_data(self, db):
        create_prediction(db, _pred())
        csv_text = export_predictions(db, export_format=ExportFormat.CSV)
        assert "image_name" in csv_text
        assert "model_name" in csv_text
        assert "test.jpg" in csv_text

    def test_csv_first_line_is_header_row(self, db):
        csv_text = export_predictions(db, export_format=ExportFormat.CSV)
        headers = csv_text.strip().split("\n")[0]
        assert "id" in headers
        assert "confidence" in headers

    def test_json_export_filters_by_model(self, db):
        create_prediction(db, _pred(model_name="model_a"))
        create_prediction(db, _pred(model_name="model_b"))
        parsed = json.loads(
            export_predictions(db, export_format=ExportFormat.JSON, model_name="model_a")
        )
        assert len(parsed) == 1
        assert parsed[0]["model_name"] == "model_a"

    def test_json_export_date_range_excludes_out_of_range(self, db):
        create_prediction(db, _pred())  # created now
        start = datetime(2000, 1, 1, tzinfo=timezone.utc)
        end = datetime(2000, 12, 31, tzinfo=timezone.utc)
        parsed = json.loads(
            export_predictions(db, export_format=ExportFormat.JSON, start_date=start, end_date=end)
        )
        assert parsed == []


# ---------------------------------------------------------------------------
# get_analytics_summary
# ---------------------------------------------------------------------------


class TestGetAnalyticsSummary:
    def test_returns_required_keys(self, db):
        result = get_analytics_summary(db)
        assert "total_predictions" in result
        assert "predictions_by_model" in result
        assert "predictions_by_class" in result
        assert "average_confidence" in result
        assert "average_confidence_by_model" in result

    def test_total_is_zero_when_empty(self, db):
        assert get_analytics_summary(db)["total_predictions"] == 0

    def test_counts_all_predictions(self, db):
        create_prediction(db, _pred())
        create_prediction(db, _pred())
        assert get_analytics_summary(db)["total_predictions"] == 2

    def test_groups_predictions_by_model(self, db):
        create_prediction(db, _pred(model_name="m_a"))
        create_prediction(db, _pred(model_name="m_a"))
        create_prediction(db, _pred(model_name="m_b"))
        result = get_analytics_summary(db)
        assert result["predictions_by_model"]["m_a"] == 2
        assert result["predictions_by_model"]["m_b"] == 1

    def test_groups_predictions_by_class(self, db):
        create_prediction(db, _pred(predicted_class="cat"))
        create_prediction(db, _pred(predicted_class="dog"))
        create_prediction(db, _pred(predicted_class="cat"))
        result = get_analytics_summary(db)
        assert result["predictions_by_class"]["cat"] == 2
        assert result["predictions_by_class"]["dog"] == 1

    def test_calculates_average_confidence(self, db):
        create_prediction(db, _pred(confidence=0.8))
        create_prediction(db, _pred(confidence=0.6))
        assert get_analytics_summary(db)["average_confidence"] == pytest.approx(0.7)

    def test_confidence_boundary_zero(self, db):
        """Confidence of exactly 0.0 must be accepted and averaged correctly."""
        create_prediction(db, _pred(confidence=0.0))
        assert get_analytics_summary(db)["average_confidence"] == pytest.approx(0.0)

    def test_confidence_boundary_one(self, db):
        """Confidence of exactly 1.0 must be accepted and averaged correctly."""
        create_prediction(db, _pred(confidence=1.0))
        assert get_analytics_summary(db)["average_confidence"] == pytest.approx(1.0)

    def test_average_confidence_per_model(self, db):
        create_prediction(db, _pred(model_name="m1", confidence=0.9))
        create_prediction(db, _pred(model_name="m2", confidence=0.5))
        result = get_analytics_summary(db)
        assert result["average_confidence_by_model"]["m1"] == pytest.approx(0.9)
        assert result["average_confidence_by_model"]["m2"] == pytest.approx(0.5)

    def _create_with_timestamp(self, db: Session, ts: datetime) -> None:
        """Insert a PredictionHistory row with an explicit timestamp."""
        from img_classifier_api.models.prediction import PredictionHistory as PH
        import json as _json

        record = PH(
            image_name="test.jpg",
            model_name="model_a",
            predicted_class="cat",
            confidence=0.75,
            probabilities=_json.dumps({"cat": 0.75, "dog": 0.25}),
            timestamp=ts,
        )
        db.add(record)
        db.commit()

    def test_start_date_excludes_older_records(self, db):
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        recent = datetime(2025, 6, 1, tzinfo=timezone.utc)
        self._create_with_timestamp(db, past)
        self._create_with_timestamp(db, recent)
        cutoff = datetime(2023, 1, 1, tzinfo=timezone.utc)
        result = get_analytics_summary(db, start_date=cutoff)
        assert result["total_predictions"] == 1

    def test_end_date_excludes_newer_records(self, db):
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        recent = datetime(2025, 6, 1, tzinfo=timezone.utc)
        self._create_with_timestamp(db, past)
        self._create_with_timestamp(db, recent)
        cutoff = datetime(2023, 1, 1, tzinfo=timezone.utc)
        result = get_analytics_summary(db, end_date=cutoff)
        assert result["total_predictions"] == 1

    def test_date_range_inclusive_boundaries(self, db):
        ts = datetime(2022, 6, 15, tzinfo=timezone.utc)
        self._create_with_timestamp(db, ts)
        result = get_analytics_summary(db, start_date=ts, end_date=ts)
        assert result["total_predictions"] == 1

    def test_date_range_excludes_all_records(self, db):
        self._create_with_timestamp(db, datetime(2020, 1, 1, tzinfo=timezone.utc))
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 12, 31, tzinfo=timezone.utc)
        result = get_analytics_summary(db, start_date=start, end_date=end)
        assert result["total_predictions"] == 0
