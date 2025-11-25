"""Tests for utility functions."""


import pytest

from img_classifier_utils import (
    clean_directory,
    ensure_directory_exists,
)


class TestFileUtils:
    """Tests for file utility functions."""

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test fixtures."""
        self.temp_dir = tmp_path

        yield


    def test_ensure_directory_exists_creates_directory(self):
        """Test that ensure_directory_exists creates a new directory."""
        new_dir = self.temp_dir / "test_dir"
        assert not new_dir.exists()

        result = ensure_directory_exists(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_directory_exists_with_existing_directory(self):
        """Test that ensure_directory_exists handles existing directories."""
        existing_dir = self.temp_dir / "existing"
        existing_dir.mkdir()
        assert existing_dir.exists()

        result = ensure_directory_exists(existing_dir)

        assert existing_dir.exists()
        assert result == existing_dir

    def test_ensure_directory_exists_creates_nested_directories(self):
        """Test that ensure_directory_exists creates nested directories."""
        nested_dir = self.temp_dir / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        result = ensure_directory_exists(nested_dir)

        assert nested_dir.exists()
        assert result == nested_dir

    def test_clean_directory_removes_files(self):
        """Test that clean_directory removes files matching pattern."""
        # Create test files
        test_dir = self.temp_dir / "clean_test"
        test_dir.mkdir()

        (test_dir / "file1.txt").write_text("test")
        (test_dir / "file2.txt").write_text("test")
        (test_dir / "file3.log").write_text("test")

        # Clean .txt files
        count = clean_directory(test_dir, "*.txt")

        assert count == 2
        assert not (test_dir / "file1.txt").exists()
        assert not (test_dir / "file2.txt").exists()
        assert (test_dir / "file3.log").exists()

    def test_clean_directory_with_default_pattern(self):
        """Test that clean_directory removes all files with default pattern."""
        # Create test files
        test_dir = self.temp_dir / "clean_all"
        test_dir.mkdir()

        (test_dir / "file1.txt").write_text("test")
        (test_dir / "file2.log").write_text("test")
        (test_dir / "file3.dat").write_text("test")

        # Clean all files
        count = clean_directory(test_dir)

        assert count == 3
        assert len(list(test_dir.iterdir())) == 0

    def test_clean_directory_with_nonexistent_directory(self):
        """Test that clean_directory handles non-existent directories."""
        nonexistent = self.temp_dir / "nonexistent"
        count = clean_directory(nonexistent)
        assert count == 0

    def test_clean_directory_preserves_subdirectories(self):
        """Test that clean_directory doesn't remove subdirectories."""
        test_dir = self.temp_dir / "preserve_dirs"
        test_dir.mkdir()

        (test_dir / "file.txt").write_text("test")
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("test")

        # Clean only top-level files
        count = clean_directory(test_dir, "*.txt")

        assert count == 1
        assert not (test_dir / "file.txt").exists()
        assert subdir.exists()
        assert (subdir / "nested.txt").exists()
