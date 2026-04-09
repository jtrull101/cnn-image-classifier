"""
Database configuration and session management.

Sets up SQLAlchemy with SQLite for storing prediction history.
Uses Alembic for database schema migrations.
"""

import os
import threading
from pathlib import Path
from typing import Generator

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Thread lock for database initialization
_init_lock = threading.Lock()
_initialized = False

# Database file location (can be overridden by DATABASE_URL env var)
DATABASE_DIR = Path(__file__).parent.parent / "data"
DATABASE_DIR.mkdir(exist_ok=True)
_DEFAULT_DATABASE_URL = f"sqlite:///{DATABASE_DIR}/predictions.db"

# Get database URL from environment or use default
DATABASE_URL = os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)

# Create base class for models
Base = declarative_base()


def _build_engine_and_session(db_url: str) -> tuple[Engine, sessionmaker]:
    eng = create_engine(db_url, connect_args={"check_same_thread": False}, echo=False)
    return eng, sessionmaker(autocommit=False, autoflush=False, bind=eng)


engine, SessionLocal = _build_engine_and_session(DATABASE_URL)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.

    Usage in FastAPI:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...

    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database using Alembic migrations.

    Runs all pending migrations to bring the database schema up to date.
    This is safe to call multiple times - Alembic tracks which migrations
    have already been applied in the alembic_version table.

    For testing with in-memory databases, falls back to Base.metadata.create_all()
    since Alembic requires a file-based database.

    Thread-safe: Uses a lock to prevent concurrent initialization attempts.
    """
    global _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        db_url = os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)
        alembic_ini_path = Path(__file__).parent.parent / "alembic.ini"

        # In-memory databases can't use Alembic migrations
        if ":memory:" in db_url:
            Base.metadata.create_all(bind=engine, checkfirst=True)
            _initialized = True
            return

        existing_tables = inspect(engine).get_table_names()

        if not alembic_ini_path.exists():
            if "prediction_history" not in existing_tables:
                Base.metadata.create_all(bind=engine, checkfirst=True)
            _initialized = True
            return

        alembic_cfg = Config(str(alembic_ini_path))
        if db_url != _DEFAULT_DATABASE_URL:
            alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        try:
            command.upgrade(alembic_cfg, "head")
            _initialized = True
        except Exception:
            if "prediction_history" not in inspect(engine).get_table_names():
                Base.metadata.create_all(bind=engine, checkfirst=True)
            _initialized = True


def reset_db_state() -> None:
    """
    Reset database initialization state and rebuild the engine for the current DATABASE_URL.

    Called by test fixtures (e.g. xdist isolation) after DATABASE_URL is updated so that
    the next init_db() call uses the correct database file.
    """
    global _initialized, engine, SessionLocal
    db_url = os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)
    with _init_lock:
        _initialized = False
        engine.dispose()
        engine, SessionLocal = _build_engine_and_session(db_url)
