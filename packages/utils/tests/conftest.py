"""Shared pytest configuration for utils package tests."""

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def isolated_tmp_dir(tmp_path_factory, worker_id) -> Iterator[Path]:
    """Provide an isolated temporary directory for each test."""
    if worker_id == "master":
        temp_dir = tmp_path_factory.mktemp("test")
    else:
        root_tmp = tmp_path_factory.getbasetemp().parent
        temp_dir = root_tmp / f"worker_{worker_id}" / f"test_{os.getpid()}"
        temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir

    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
