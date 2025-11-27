"""Tests for utility functions."""

import pytest

from img_classifier_utils import (
    clean_directory,
    ensure_directory_exists,
)


@pytest.mark.unit
class TestFileUtils:
    """Tests for file utility functions."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir

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
        assert subdir.exists()
        assert (subdir / "nested.txt").exists()


@pytest.mark.unit
class TestFileUtilsEdgeCases:
    """Edge case tests for file utility functions."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        yield

    def test_ensure_directory_exists_with_pathlib_path(self):
        """Test ensure_directory_exists with pathlib.Path object."""
        from pathlib import Path

        new_dir = Path(self.temp_dir) / "pathlib_test"
        result = ensure_directory_exists(new_dir)

        assert new_dir.exists()
        assert result == new_dir

    def test_ensure_directory_exists_deeply_nested(self):
        """Test ensure_directory_exists with very deep nesting."""
        from pathlib import Path

        temp_path = Path(self.temp_dir)
        deep_dir = temp_path / "l1" / "l2" / "l3" / "l4" / "l5" / "l6" / "l7"
        result = ensure_directory_exists(deep_dir)

        assert deep_dir.exists()
        assert all(
            parent.exists()
            for parent in [
                temp_path / "l1",
                temp_path / "l1" / "l2",
                temp_path / "l1" / "l2" / "l3",
            ]
        )

    def test_ensure_directory_exists_with_file_conflict(self):
        """Test ensure_directory_exists when a file exists with same name."""
        from pathlib import Path

        # Create a file
        temp_path = Path(self.temp_dir)
        conflict_path = temp_path / "conflict"
        conflict_path.write_text("test")

        # Try to create directory with same name should raise error
        with pytest.raises(Exception):
            ensure_directory_exists(conflict_path)

    def test_ensure_directory_exists_multiple_calls(self):
        """Test that multiple calls to ensure_directory_exists are safe."""
        from pathlib import Path

        temp_path = Path(self.temp_dir)
        test_dir = temp_path / "multiple"

        result1 = ensure_directory_exists(test_dir)
        result2 = ensure_directory_exists(test_dir)
        result3 = ensure_directory_exists(test_dir)

        assert result1 == result2 == result3
        assert test_dir.exists()

    def test_clean_directory_with_wildcard_patterns(self):
        """Test clean_directory with various wildcard patterns."""
        test_dir = self.temp_dir / "wildcards"
        test_dir.mkdir()

        # Create various files
        (test_dir / "image1.jpg").write_text("test")
        (test_dir / "image2.png").write_text("test")
        (test_dir / "document.pdf").write_text("test")
        (test_dir / "archive.tar.gz").write_text("test")
        (test_dir / "README.md").write_text("test")

        # Clean only image files
        count = clean_directory(test_dir, "*.jpg")
        assert count == 1
        assert not (test_dir / "image1.jpg").exists()
        assert (test_dir / "image2.png").exists()

    def test_clean_directory_with_multiple_extensions(self):
        """Test clean_directory by calling multiple times with different patterns."""
        test_dir = self.temp_dir / "multi_ext"
        test_dir.mkdir()

        (test_dir / "file1.txt").write_text("test")
        (test_dir / "file2.log").write_text("test")
        (test_dir / "file3.tmp").write_text("test")
        (test_dir / "file4.dat").write_text("test")

        # Clean txt files
        count1 = clean_directory(test_dir, "*.txt")
        assert count1 == 1

        # Clean log files
        count2 = clean_directory(test_dir, "*.log")
        assert count2 == 1

        # Verify remaining files
        remaining = list(test_dir.iterdir())
        assert len(remaining) == 2

    def test_clean_directory_empty_directory(self):
        """Test clean_directory on empty directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()

        count = clean_directory(empty_dir)
        assert count == 0
        assert empty_dir.exists()

    def test_clean_directory_with_hidden_files(self):
        """Test clean_directory with hidden files (starting with dot)."""
        test_dir = self.temp_dir / "hidden_files"
        test_dir.mkdir()

        (test_dir / "visible.txt").write_text("test")
        (test_dir / ".hidden").write_text("test")
        (test_dir / ".config").write_text("test")

        # Clean all files
        count = clean_directory(test_dir, "*")

        # Should remove all files including hidden ones
        assert count >= 1  # At least visible file
        assert not (test_dir / "visible.txt").exists()

    def test_clean_directory_preserves_nested_structure(self):
        """Test clean_directory preserves nested directory structure."""
        test_dir = self.temp_dir / "nested_preserve"
        test_dir.mkdir()

        # Create nested structure
        (test_dir / "file1.txt").write_text("test")
        subdir1 = test_dir / "sub1"
        subdir1.mkdir()
        (subdir1 / "file2.txt").write_text("test")
        subdir2 = subdir1 / "sub2"
        subdir2.mkdir()
        (subdir2 / "file3.txt").write_text("test")

        # Clean only top-level
        count = clean_directory(test_dir, "*.txt")

        assert count == 1  # Only top-level file
        assert subdir1.exists()
        assert subdir2.exists()
        assert (subdir1 / "file2.txt").exists()
        assert (subdir2 / "file3.txt").exists()

    def test_clean_directory_with_special_characters_in_filenames(self):
        """Test clean_directory with special characters in filenames."""
        test_dir = self.temp_dir / "special_chars"
        test_dir.mkdir()

        (test_dir / "file-1.txt").write_text("test")
        (test_dir / "file_2.txt").write_text("test")
        (test_dir / "file 3.txt").write_text("test")
        (test_dir / "file.backup.txt").write_text("test")

        count = clean_directory(test_dir, "*.txt")

        assert count == 4
        assert len(list(test_dir.iterdir())) == 0

    def test_clean_directory_with_read_only_files(self):
        """Test clean_directory with read-only files (platform dependent)."""
        import os
        import stat

        test_dir = self.temp_dir / "readonly"
        test_dir.mkdir()

        # Create a read-only file
        readonly_file = test_dir / "readonly.txt"
        readonly_file.write_text("test")

        # Make it read-only
        os.chmod(readonly_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        try:
            count = clean_directory(test_dir, "*.txt")
            # May or may not succeed depending on platform and permissions
            # Just verify it doesn't crash
            assert isinstance(count, int)
        finally:
            # Restore write permissions for cleanup
            try:
                os.chmod(readonly_file, stat.S_IWRITE | stat.S_IREAD)
            except Exception:
                pass

    def test_ensure_directory_exists_with_unicode_name(self):
        """Test ensure_directory_exists with unicode characters in path."""
        from pathlib import Path

        temp_path = Path(self.temp_dir)
        unicode_dir = temp_path / "测试目录" / "ディレクトリ" / "каталог"
        result = ensure_directory_exists(unicode_dir)

        assert unicode_dir.exists()
        assert result == unicode_dir

    def test_clean_directory_with_large_number_of_files(self):
        """Test clean_directory with many files."""
        test_dir = self.temp_dir / "many_files"
        test_dir.mkdir()

        # Create 100 files
        for i in range(100):
            (test_dir / f"file_{i}.txt").write_text(f"content_{i}")

        count = clean_directory(test_dir, "*.txt")

        assert count == 100
        assert len(list(test_dir.iterdir())) == 0

    def test_clean_directory_with_symbolic_links(self):
        """Test clean_directory behavior with symbolic links (if supported)."""
        import platform

        if platform.system() == "Windows":
            pytest.skip("Symbolic links require special permissions on Windows")

        test_dir = self.temp_dir / "symlinks"
        test_dir.mkdir()

        # Create a real file and a symlink
        real_file = test_dir / "real.txt"
        real_file.write_text("real content")

        target_file = self.temp_dir / "target.txt"
        target_file.write_text("target content")

        try:
            link_file = test_dir / "link.txt"
            link_file.symlink_to(target_file)

            count = clean_directory(test_dir, "*.txt")

            # Should remove both real file and symlink
            assert count == 2
            assert target_file.exists()  # Target should still exist
        except (OSError, NotImplementedError):
            pytest.skip("Symbolic links not supported")

    def test_ensure_directory_exists_returns_path_object(self):
        """Test that ensure_directory_exists returns Path object."""
        from pathlib import Path

        temp_path = Path(self.temp_dir)

        # Test with Path
        result = ensure_directory_exists(temp_path / "test1")
        assert isinstance(result, Path)
        assert result.exists()

        # Test returns same path
        result2 = ensure_directory_exists(temp_path / "test2")
        assert isinstance(result2, Path)
        assert result2.name == "test2"

    def test_clean_directory_return_value_accuracy(self):
        """Test that clean_directory returns accurate count."""
        test_dir = self.temp_dir / "count_test"
        test_dir.mkdir()

        # Create known number of files
        expected_count = 7
        for i in range(expected_count):
            (test_dir / f"file{i}.tmp").write_text("test")

        count = clean_directory(test_dir, "*.tmp")

        assert count == expected_count
        assert len(list(test_dir.glob("*.tmp"))) == 0
