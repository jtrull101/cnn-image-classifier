#!/usr/bin/env python3
"""Initialize the database using Alembic migrations.

This script can be used to manually initialize the database or as part of deployment automation.
It's safe to run multiple times - Alembic will only apply migrations that haven't been run yet.
"""

import sys
from pathlib import Path


# Add parent directory to path so we can import from img_classifier_api
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    from img_classifier_api.database import init_db

    print("Initializing database with Alembic migrations...")
    try:
        init_db()
        print("✓ Database initialized successfully!")
        print("  Database location: apps/api/data/predictions.db")

        # Show current version
        try:
            from alembic import command
            from alembic.config import Config

            alembic_cfg = Config(str(Path(__file__).parent.parent / "alembic.ini"))
            print("\nCurrent migration version:")
            command.current(alembic_cfg, verbose=True)
        except Exception:
            pass  # Ignore if we can't get version info

    except Exception as e:
        print(f"✗ Failed to initialize database: {e}", file=sys.stderr)
        sys.exit(1)
