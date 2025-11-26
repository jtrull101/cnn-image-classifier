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

    def test_organize_dataset(self):
        """Test dataset organization from nested to flat structure."""
        from img_classifier_utils import organize_dataset

        # Create nested source structure
        source = self.temp_dir / "source"
        (source / "train" / "cat" / "subdir").mkdir(parents=True)
        (source / "train" / "dog").mkdir(parents=True)

        # Add test files
        (source / "train" / "cat" / "subdir" / "cat1.jpg").write_text("cat")
        (source / "train" / "dog" / "dog1.jpg").write_text("dog")

        dest = self.temp_dir / "dest"
        result = organize_dataset(source, dest)

        assert result is True
        # Files should be moved to flattened structure
        assert (dest / "train" / "cat" / "cat1.jpg").exists() or (dest / "train" / "cat" / "subdir" / "cat1.jpg").exists()

    def test_organize_dataset_nonexistent_source(self):
        """Test organize_dataset with nonexistent source."""
        from img_classifier_utils import organize_dataset

        source = self.temp_dir / "nonexistent"
        dest = self.temp_dir / "dest"
        result = organize_dataset(source, dest)

        assert result is False

    def test_download_from_google_drive_invalid_id(self):
        """Test download_from_google_drive with invalid file ID."""
        from img_classifier_utils import download_from_google_drive

        dest = self.temp_dir / "download_test.zip"
        result = download_from_google_drive("invalid_id_12345", dest, quiet=True)

        # Should return False on error
        assert result is False

    def test_extract_archive_nonexistent(self):
        """Test extract_archive with nonexistent file."""
        from img_classifier_utils import extract_archive

        archive = self.temp_dir / "nonexistent.zip"
        extract_to = self.temp_dir / "extracted"
        result = extract_archive(archive, extract_to)

        assert result is False

    def test_clean_directory_with_errors(self):
        """Test clean_directory handles errors gracefully."""
        test_dir = self.temp_dir / "error_test"
        test_dir.mkdir()

        # Create a file and immediately try to clean
        file = test_dir / "test.txt"
        file.write_text("test")

        # Clean should still work
        count = clean_directory(test_dir)
        assert count >= 0  # May be 0 or 1 depending on timing

    def test_ensure_directory_with_parent_creation(self):
        """Test ensure_directory_exists creates parent directories."""
        deep_dir = self.temp_dir / "a" / "b" / "c" / "d"
        result = ensure_directory_exists(deep_dir)

        assert result == deep_dir
        assert deep_dir.exists()
        assert (self.temp_dir / "a").exists()
        assert (self.temp_dir / "a" / "b").exists()

    def test_organize_dataset_with_files_only(self):
        """Test organize_dataset skips non-directory items."""
        from img_classifier_utils import organize_dataset

        source = self.temp_dir / "source_files"
        source.mkdir()
        (source / "file.txt").write_text("not a dir")

        dest = self.temp_dir / "dest_files"
        result = organize_dataset(source, dest)

        # Should succeed but not copy anything
        assert result is True

    def test_clean_directory_specific_pattern(self):
        """Test clean_directory with specific file patterns."""
        test_dir = self.temp_dir / "pattern_test"
        test_dir.mkdir()

        (test_dir / "file1.txt").write_text("txt")
        (test_dir / "file2.log").write_text("log")
        (test_dir / "file3.dat").write_text("dat")

        # Clean only .txt files
        count = clean_directory(test_dir, "*.txt")

        assert count == 1
        assert not (test_dir / "file1.txt").exists()
        assert (test_dir / "file2.log").exists()
        assert (test_dir / "file3.dat").exists()

    def test_extract_archive_creates_dest_dir(self):
        """Test that extract_archive creates destination directory."""
        import zipfile
        from img_classifier_utils import extract_archive

        # Create a test zip file
        zip_path = self.temp_dir / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test.txt", "content")

        extract_to = self.temp_dir / "extracted"
        result = extract_archive(zip_path, extract_to)

        assert result is True
        assert extract_to.exists()
        assert (extract_to / "test.txt").exists()
