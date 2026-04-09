"""
Database configuration and session management.

Sets up SQLAlchemy with SQLite for storing prediction history.
Uses Alembic for database schema migrations.
"""

import os
from pathlib import Path
from typing import Generator

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, declarative_base, sessionmaker


# Database file location (can be overridden by DATABASE_URL env var)
DATABASE_DIR = Path(__file__).parent.parent / "data"
DATABASE_DIR.mkdir(exist_ok=True)
_DEFAULT_DATABASE_URL = f"sqlite:///{DATABASE_DIR}/predictions.db"

# Get database URL from environment or use default
DATABASE_URL = os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False,  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


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
    """
    # Get path to alembic.ini
    alembic_ini_path = Path(__file__).parent.parent / "alembic.ini"

    # Check if using in-memory database (for testing)
    db_url = os.environ.get("DATABASE_URL")
    if db_url and ":memory:" in db_url:
        # In-memory databases can't use Alembic, use create_all instead
        Base.metadata.create_all(bind=engine, checkfirst=True)
        return

    # Check if database already has tables (migration already ran)
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    if not alembic_ini_path.exists():
        # Fallback to create_all for environments without alembic.ini
        if "prediction_history" not in existing_tables:
            Base.metadata.create_all(bind=engine, checkfirst=True)
        return

    # Use Alembic for migrations
    alembic_cfg = Config(str(alembic_ini_path))

    # Override the database URL if DATABASE_URL env var is set
    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    # Run migrations to head (safe to call multiple times)
    try:
        command.upgrade(alembic_cfg, "head")
    except Exception:
        # If Alembic fails (e.g., concurrent access), check if tables exist
        # and fall back to create_all if needed
        inspector = inspect(engine)
        if "prediction_history" not in inspector.get_table_names():
            # Only create if table doesn't exist (race condition protection)
            Base.metadata.create_all(bind=engine, checkfirst=True)
        # If tables exist, the migration already ran successfully elsewhere
