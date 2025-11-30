"""Shared pytest configuration for data package tests."""

import sys
from pathlib import Path

# Reuse the repository-level pytest configuration (markers, fixtures).
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import (  # noqa: E402,F401
    isolated_tmp_dir,
    mock_config,
    mock_config_paths,
    mock_data_loader,
    mock_model,
    test_data_dir,
)
