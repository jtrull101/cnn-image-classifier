"""Unit tests for database session management and table initialisation."""

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_in_memory_engine():
    """Return a fresh in-memory SQLite engine for isolated testing."""
    return create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    """init_db() creates all ORM-mapped tables on the bound engine."""

    def test_creates_prediction_history_table(self):
        """After init_db() the prediction_history table must exist."""
        from img_classifier_api.models import prediction  # noqa: F401 — registers models

        engine = _make_in_memory_engine()
        with patch("img_classifier_api.database.engine", engine):
            from img_classifier_api.database import init_db

            init_db()

        table_names = inspect(engine).get_table_names()
        assert "prediction_history" in table_names

    def test_init_db_is_idempotent(self):
        """Calling init_db() twice must not raise."""
        from img_classifier_api.models import prediction  # noqa: F401

        engine = _make_in_memory_engine()
        with patch("img_classifier_api.database.engine", engine):
            from img_classifier_api.database import init_db

            init_db()
            init_db()  # second call — should not raise

        assert "prediction_history" in inspect(engine).get_table_names()

    def test_tables_have_expected_columns(self):
        """prediction_history table must have the id and model_name columns."""
        from img_classifier_api.models import prediction  # noqa: F401

        engine = _make_in_memory_engine()
        with patch("img_classifier_api.database.engine", engine):
            from img_classifier_api.database import init_db

            init_db()

        cols = {c["name"] for c in inspect(engine).get_columns("prediction_history")}
        assert "id" in cols
        assert "model_name" in cols
        assert "predicted_class" in cols
        assert "confidence" in cols


# ---------------------------------------------------------------------------
# get_db
# ---------------------------------------------------------------------------


class TestGetDb:
    """get_db() yields a SQLAlchemy Session and closes it after use."""

    def test_yields_session_instance(self):
        """get_db() must yield a SQLAlchemy Session."""
        from img_classifier_api.database import get_db

        gen = get_db()
        session = next(gen)
        try:
            assert isinstance(session, Session)
        finally:
            try:
                next(gen)
            except StopIteration:
                pass

    def test_session_is_closed_after_generator_exhausted(self):
        """session.close() must be called once the generator is exhausted."""
        from unittest.mock import patch

        from img_classifier_api.database import get_db

        gen = get_db()
        session = next(gen)

        with patch.object(session, "close", wraps=session.close) as mock_close:
            try:
                next(gen)
            except StopIteration:
                pass

            mock_close.assert_called_once()

    def test_session_closed_on_exception(self):
        """session.close() must be called via the finally block even when a route raises."""
        from unittest.mock import patch

        from img_classifier_api.database import get_db

        gen = get_db()
        session = next(gen)

        with patch.object(session, "close", wraps=session.close) as mock_close:
            # Simulate an error in the route handler by throwing into the generator
            try:
                gen.throw(RuntimeError("simulated route error"))
            except RuntimeError:
                pass

            mock_close.assert_called_once()

    def test_each_call_returns_new_session(self):
        """Two successive calls to get_db() must return distinct session objects."""
        from img_classifier_api.database import get_db

        gen1 = get_db()
        gen2 = get_db()
        s1 = next(gen1)
        s2 = next(gen2)

        assert s1 is not s2

        for gen in (gen1, gen2):
            try:
                next(gen)
            except StopIteration:
                pass

    def test_session_can_execute_query(self):
        """A session yielded by get_db() must be able to run a trivial query."""
        from img_classifier_api.database import get_db

        gen = get_db()
        session = next(gen)
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1

        try:
            next(gen)
        except StopIteration:
            pass
