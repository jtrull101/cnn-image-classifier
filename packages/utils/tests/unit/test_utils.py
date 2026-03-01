"""Comprehensive tests for utility functions."""

import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from img_classifier_utils import (
    clean_directory,
    download_from_google_drive,
    ensure_directory_exists,
    extract_archive,
    organize_dataset,
)

pytestmark = pytest.mark.unit


class TestFileUtils:
    """Tests for file utility functions."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir

        yield

    @pytest.mark.smoke
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

    @pytest.mark.smoke
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

        count = clean_directory(test_dir)

        assert count == 1
        assert subdir.exists()
        assert subdir.is_dir()

    def test_clean_directory_specific_extension(self):
        """Test cleaning only specific file extensions."""
        test_dir = self.temp_dir / "specific_ext"
        test_dir.mkdir()

        (test_dir / "file1.jpg").write_text("test")
        (test_dir / "file2.png").write_text("test")
        (test_dir / "file3.jpg").write_text("test")

        count = clean_directory(test_dir, "*.jpg")

        assert count == 2
        assert (test_dir / "file2.png").exists()


class TestDownloadFromGoogleDrive:
    """Tests for Google Drive download functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        yield

    @patch("img_classifier_utils.file_utils.gdown.download")
    def test_download_success(self, mock_download):
        """Test successful download from Google Drive."""
        mock_download.return_value = "downloaded_file.zip"

        destination = self.temp_dir / "data" / "file.zip"
        result = download_from_google_drive("test_file_id", destination)

        assert result is True
        mock_download.assert_called_once()
        assert destination.parent.exists()

    @patch("img_classifier_utils.file_utils.gdown.download")
    def test_download_creates_parent_directory(self, mock_download):
        """Test that download creates parent directories."""
        mock_download.return_value = "downloaded_file.zip"

        destination = self.temp_dir / "level1" / "level2" / "file.zip"
        download_from_google_drive("test_file_id", destination)

        assert destination.parent.exists()

    @patch("img_classifier_utils.file_utils.gdown.download")
    def test_download_with_quiet_flag(self, mock_download):
        """Test download with quiet flag."""
        mock_download.return_value = "downloaded_file.zip"

        destination = self.temp_dir / "file.zip"
        download_from_google_drive("test_file_id", destination, quiet=True)

        call_args = mock_download.call_args
        assert call_args.kwargs["quiet"] is True

    @patch("img_classifier_utils.file_utils.gdown.download")
    def test_download_failure(self, mock_download):
        """Test handling of download failure."""
        mock_download.side_effect = Exception("Network error")

        destination = self.temp_dir / "file.zip"
        result = download_from_google_drive("test_file_id", destination)

        assert result is False

    @patch("img_classifier_utils.file_utils.gdown.download")
    def test_download_url_format(self, mock_download):
        """Test that correct URL is constructed."""
        mock_download.return_value = "downloaded_file.zip"

        destination = self.temp_dir / "file.zip"
        download_from_google_drive("ABC123", destination)

        call_args = mock_download.call_args
        expected_url = "https://docs.google.com/uc?id=ABC123"
        assert call_args.args[0] == expected_url


class TestExtractArchive:
    """Tests for archive extraction functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        yield

    def test_extract_zip_file(self):
        """Test extracting a zip file."""
        # Create a test zip file
        zip_path = self.temp_dir / "test.zip"
        extract_to = self.temp_dir / "extracted"

        # Create test files to zip
        test_data_dir = self.temp_dir / "test_data"
        test_data_dir.mkdir()
        (test_data_dir / "file1.txt").write_text("content1")
        (test_data_dir / "file2.txt").write_text("content2")

        # Create zip
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(test_data_dir / "file1.txt", "file1.txt")
            zf.write(test_data_dir / "file2.txt", "file2.txt")

        result = extract_archive(zip_path, extract_to)

        assert result is True
        assert extract_to.exists()
        assert (extract_to / "file1.txt").exists()
        assert (extract_to / "file2.txt").exists()

    def test_extract_creates_directory(self):
        """Test that extract creates target directory."""
        zip_path = self.temp_dir / "test.zip"
        extract_to = self.temp_dir / "deep" / "nested" / "dir"

        # Create empty zip
        with zipfile.ZipFile(zip_path, "w"):
            pass

        extract_archive(zip_path, extract_to)

        assert extract_to.exists()

    def test_extract_nonexistent_archive(self):
        """Test extracting non-existent archive."""
        nonexistent = self.temp_dir / "nonexistent.zip"
        extract_to = self.temp_dir / "extracted"

        result = extract_archive(nonexistent, extract_to)

        assert result is False

    @patch("img_classifier_utils.file_utils.Archive")
    def test_extract_non_zip_archive(self, mock_archive):
        """Test extracting non-zip archive (using pyunpack)."""
        # Create a dummy file with non-zip extension
        tar_path = self.temp_dir / "test.tar.gz"
        tar_path.write_text("dummy content")
        extract_to = self.temp_dir / "extracted"

        mock_archive_instance = MagicMock()
        mock_archive.return_value = mock_archive_instance

        result = extract_archive(tar_path, extract_to)

        assert result is True
        mock_archive.assert_called_once_with(str(tar_path))
        mock_archive_instance.extractall.assert_called_once_with(str(extract_to))


class TestOrganizeDataset:
    """Tests for dataset organization functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        yield

    def test_organize_nested_structure(self):
        """Test organizing nested dataset structure."""
        # Create source structure: source/train/class1/file.jpg
        source = self.temp_dir / "source"
        train_dir = source / "train" / "class1"
        train_dir.mkdir(parents=True)
        (train_dir / "file1.jpg").write_text("image1")
        (train_dir / "file2.jpg").write_text("image2")

        dest = self.temp_dir / "dest"

        result = organize_dataset(source, dest)

        assert result is True
        assert (dest / "train" / "class1" / "file1.jpg").exists()
        assert (dest / "train" / "class1" / "file2.jpg").exists()

    def test_organize_multiple_classes(self):
        """Test organizing multiple classes."""
        source = self.temp_dir / "source"

        # Create multiple classes
        for split in ["train", "test"]:
            for cls in ["class1", "class2"]:
                cls_dir = source / split / cls
                cls_dir.mkdir(parents=True)
                (cls_dir / f"{split}_{cls}.jpg").write_text("image")

        dest = self.temp_dir / "dest"
        organize_dataset(source, dest)

        assert (dest / "train" / "class1" / "train_class1.jpg").exists()
        assert (dest / "train" / "class2" / "train_class2.jpg").exists()
        assert (dest / "test" / "class1" / "test_class1.jpg").exists()
        assert (dest / "test" / "class2" / "test_class2.jpg").exists()

    def test_organize_nonexistent_source(self):
        """Test organizing from non-existent source."""
        source = self.temp_dir / "nonexistent"
        dest = self.temp_dir / "dest"

        result = organize_dataset(source, dest)

        assert result is False

    def test_organize_skips_existing_files(self):
        """Test that organize doesn't overwrite existing files."""
        source = self.temp_dir / "source"
        train_dir = source / "train" / "class1"
        train_dir.mkdir(parents=True)
        (train_dir / "file.jpg").write_text("source_content")

        dest = self.temp_dir / "dest"
        dest_file = dest / "train" / "class1" / "file.jpg"
        dest_file.parent.mkdir(parents=True)
        dest_file.write_text("existing_content")

        organize_dataset(source, dest)

        # Should not overwrite
        assert dest_file.read_text() == "existing_content"

    def test_organize_ignores_non_directories(self):
        """Test that organize ignores non-directory files."""
        source = self.temp_dir / "source"
        source.mkdir()

        # Create file at split level (should be ignored)
        (source / "readme.txt").write_text("readme")

        # Create valid structure
        train_dir = source / "train" / "class1"
        train_dir.mkdir(parents=True)
        (train_dir / "image.jpg").write_text("image")

        dest = self.temp_dir / "dest"
        result = organize_dataset(source, dest)

        assert result is True
        assert (dest / "train" / "class1" / "image.jpg").exists()

    def test_organize_handles_empty_categories(self):
        """Test organizing with empty category directories."""
        source = self.temp_dir / "source"
        train_dir = source / "train" / "empty_class"
        train_dir.mkdir(parents=True)

        dest = self.temp_dir / "dest"
        result = organize_dataset(source, dest)

        assert result is True

    def test_organize_exception_handling(self):
        """Test that organize handles exceptions gracefully."""
        source = self.temp_dir / "source"
        source.mkdir()

        # Create invalid structure that might cause issues
        train_dir = source / "train" / "class1"
        train_dir.mkdir(parents=True)
        (train_dir / "image.jpg").write_text("image")

        dest = self.temp_dir / "dest"

        # Mock shutil.copy2 to raise exception
        with patch("shutil.copy2", side_effect=Exception("Copy failed")):
            result = organize_dataset(source, dest)

            assert result is False


class TestUtilsEdgeCases:
    """Edge case tests for utility functions."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        yield

    def test_clean_directory_empty_dir(self):
        """Test cleaning empty directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()

        count = clean_directory(empty_dir)

        assert count == 0

    def test_clean_directory_with_hidden_files(self):
        """Test cleaning directory with hidden files."""
        test_dir = self.temp_dir / "hidden"
        test_dir.mkdir()

        (test_dir / ".hidden").write_text("hidden")
        (test_dir / "visible.txt").write_text("visible")

        count = clean_directory(test_dir)

        assert count == 2

    def test_ensure_directory_exists_with_file_conflict(self):
        """Test ensure_directory_exists when a file exists with same name."""
        file_path = self.temp_dir / "conflict"
        file_path.write_text("file")

        # This should raise an error as file exists
        with pytest.raises(Exception):
            ensure_directory_exists(file_path)

    def test_extract_zip_with_nested_structure(self):
        """Test extracting zip with nested folder structure."""
        zip_path = self.temp_dir / "nested.zip"
        extract_to = self.temp_dir / "extracted"

        # Create nested structure
        nested_dir = self.temp_dir / "nested" / "subdir"
        nested_dir.mkdir(parents=True)
        (nested_dir / "file.txt").write_text("content")

        # Create zip with nested structure
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(nested_dir / "file.txt", "nested/subdir/file.txt")

        result = extract_archive(zip_path, extract_to)

        assert result is True
        assert (extract_to / "nested" / "subdir" / "file.txt").exists()


class TestFileUtilsEdgeCases:
    """Edge case tests for file utility functions."""

    @pytest.fixture(autouse=True)
    def setup_method(self, isolated_tmp_dir):
        """Set up test fixtures."""
        self.temp_dir = isolated_tmp_dir
        yield

    def test_ensure_directory_exists_with_pathlib_path(self):
        """Test ensure_directory_exists with pathlib.Path object."""

        new_dir = Path(self.temp_dir) / "pathlib_test"
        result = ensure_directory_exists(new_dir)

        assert new_dir.exists()
        assert result == new_dir

    def test_ensure_directory_exists_deeply_nested(self):
        """Test ensure_directory_exists with very deep nesting."""

        temp_path = Path(self.temp_dir)
        deep_dir = temp_path / "l1" / "l2" / "l3" / "l4" / "l5" / "l6" / "l7"
        ensure_directory_exists(deep_dir)

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

        # Create a file
        temp_path = Path(self.temp_dir)
        conflict_path = temp_path / "conflict"
        conflict_path.write_text("test")

        # Try to create directory with same name should raise error
        with pytest.raises(Exception):
            ensure_directory_exists(conflict_path)

    def test_ensure_directory_exists_multiple_calls(self):
        """Test that multiple calls to ensure_directory_exists are safe."""

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

    def test_organize_dataset_with_non_directory_in_split(self):
        """Test organize_dataset skips non-directory items in split directory."""
        source = self.temp_dir / "source_with_files"
        dest = self.temp_dir / "dest_files"

        train_dir = source / "train"
        train_dir.mkdir(parents=True)

        # Create a file in train directory (not a subdirectory)
        (train_dir / "random_file.txt").write_text("should be skipped")

        # Create normal class directory
        class_dir = train_dir / "class1"
        class_dir.mkdir()
        (class_dir / "img.jpg").write_text("image")

        result = organize_dataset(source, dest)

        assert result is True
        # File should not be copied
        assert not (dest / "train" / "random_file.txt").exists()
        # But class directory should be copied
        assert (dest / "train" / "class1" / "img.jpg").exists()

    def test_organize_dataset_with_non_file_in_category(self):
        """Test organize_dataset skips non-file items in category directory."""
        source = self.temp_dir / "source_with_dir"
        dest = self.temp_dir / "dest_dir"

        class_dir = source / "train" / "class1"
        class_dir.mkdir(parents=True)

        # Create a subdirectory in class directory (should be skipped)
        (class_dir / "subdir").mkdir()
        (class_dir / "subdir" / "nested.jpg").write_text("nested image")

        # Create normal file
        (class_dir / "img.jpg").write_text("image")

        result = organize_dataset(source, dest)

        assert result is True
        # Subdirectory should not be copied as a file
        assert not (dest / "train" / "class1" / "subdir").is_file()
        # But normal file should be copied
        assert (dest / "train" / "class1" / "img.jpg").exists()

    def test_clean_directory_file_removal_error(self):
        """Test clean_directory handles file removal errors gracefully."""
        test_dir = self.temp_dir / "error_test"
        test_dir.mkdir()

        # Create a file
        test_file = test_dir / "file.txt"
        test_file.write_text("content")

        # Make file read-only to cause permission error
        import os
        import stat

        os.chmod(test_file, stat.S_IRUSR)

        try:
            # Should handle error and continue
            count = clean_directory(test_dir, "*.txt")
            # On some systems it may succeed, on others it may fail
            assert count >= 0
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(test_file, stat.S_IWUSR | stat.S_IRUSR)
            except Exception:
                pass
