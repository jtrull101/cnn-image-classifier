"""Shared pytest configuration for models package tests."""

import os
import shutil
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture(scope="function")
def isolated_tmp_dir(tmp_path_factory, worker_id) -> Iterator[Path]:
    """
    Provide an isolated temporary directory for each test, safe for parallel execution.

    This fixture uses pytest-xdist's worker_id to ensure each worker process
    gets its own isolated directory space.
    """
    if worker_id == "master":
        # Single process - use standard tmp_path
        temp_dir = tmp_path_factory.mktemp("test")
    else:
        # Multiple workers - create worker-specific directories
        root_tmp = tmp_path_factory.getbasetemp().parent
        temp_dir = root_tmp / f"worker_{worker_id}" / f"test_{os.getpid()}"
        temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
