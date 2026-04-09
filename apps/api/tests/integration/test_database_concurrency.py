"""
Test to verify database initialization race condition is fixed.

This test simulates parallel workers trying to initialize the database simultaneously.
"""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

# Mark all tests in this module as integration and serial
# Serial prevents pytest-xdist from running these in parallel with other tests
pytestmark = [pytest.mark.integration, pytest.mark.serial]


@pytest.mark.timeout(30)  # Fail fast if test hangs
def test_concurrent_database_initialization():
    """Test that multiple workers can initialize the database without race conditions."""
    from img_classifier_api.database import init_db, reset_db_state

    # Create a temporary database file
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_concurrent.db"
        db_url = f"sqlite:///{db_file}"

        # Set environment variable
        original_url = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = db_url

        try:
            # Reset initialization state to test concurrent initialization
            reset_db_state()

            # Simulate 5 workers all trying to initialize at once
            num_workers = 5
            results = []

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(init_db) for _ in range(num_workers)]

                # Add timeout to as_completed to prevent hanging
                for future in as_completed(futures, timeout=20):
                    try:
                        future.result(timeout=5)  # Individual future timeout
                        results.append("success")
                    except Exception as e:
                        results.append(f"error: {e}")

            # All workers should succeed (no race condition errors)
            assert all(r == "success" for r in results), (
                f"Expected all workers to succeed, got: {results}"
            )

            # Verify database file was created
            assert db_file.exists(), "Database file should exist"

            # Verify table was created
            from sqlalchemy import inspect, create_engine

            engine = create_engine(db_url)
            try:
                inspector = inspect(engine)
                tables = inspector.get_table_names()
                assert "prediction_history" in tables, "Table should be created"
                assert "alembic_version" in tables, "Alembic version table should exist"
            finally:
                engine.dispose()  # Properly close connections

        finally:
            # Restore original environment
            if original_url:
                os.environ["DATABASE_URL"] = original_url
            elif "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]

            # Reset state for other tests
            reset_db_state()


@pytest.mark.timeout(10)
def test_worker_database_isolation(tmp_path):
    """Test that the fixture creates isolated databases for each worker."""
    from img_classifier_api.database import init_db, reset_db_state
    from sqlalchemy import create_engine, inspect
    from unittest.mock import patch

    worker_id = "test_worker_1"
    db_file = tmp_path / f"predictions_worker_{worker_id}.db"
    db_url = f"sqlite:///{db_file}"

    # Verify the URL pattern works
    assert "worker" in db_url
    assert worker_id in db_url

    # Create database with this URL
    original_url = os.environ.get("DATABASE_URL")

    try:
        # Create engine for this specific database
        test_engine = create_engine(db_url, connect_args={"check_same_thread": False})

        # Reset state and patch engine to use our test database
        reset_db_state()
        os.environ["DATABASE_URL"] = db_url

        with patch("img_classifier_api.database.engine", test_engine):
            init_db()

        # Verify file was created
        assert db_file.exists(), f"Database file {db_file} should exist"

        # Verify table was created
        inspector = inspect(test_engine)
        tables = inspector.get_table_names()
        assert "prediction_history" in tables, "Table should be created in isolated database"

        test_engine.dispose()

    finally:
        if original_url:
            os.environ["DATABASE_URL"] = original_url
        elif "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        reset_db_state()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
