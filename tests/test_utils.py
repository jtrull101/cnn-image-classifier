"""Tests for utility functions."""

import unittest
from pathlib import Path
import tempfile
import shutil

from src.alz_mri_cnn.utils import (
    ensure_directory_exists,
    clean_directory,
)


class TestFileUtils(unittest.TestCase):
    """Tests for file utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_ensure_directory_exists_creates_directory(self):
        """Test that ensure_directory_exists creates a new directory."""
        new_dir = self.temp_dir / "test_dir"
        self.assertFalse(new_dir.exists())
        
        result = ensure_directory_exists(new_dir)
        
        self.assertTrue(new_dir.exists())
        self.assertTrue(new_dir.is_dir())
        self.assertEqual(result, new_dir)

    def test_ensure_directory_exists_with_existing_directory(self):
        """Test that ensure_directory_exists handles existing directories."""
        existing_dir = self.temp_dir / "existing"
        existing_dir.mkdir()
        self.assertTrue(existing_dir.exists())
        
        result = ensure_directory_exists(existing_dir)
        
        self.assertTrue(existing_dir.exists())
        self.assertEqual(result, existing_dir)

    def test_ensure_directory_exists_creates_nested_directories(self):
        """Test that ensure_directory_exists creates nested directories."""
        nested_dir = self.temp_dir / "level1" / "level2" / "level3"
        self.assertFalse(nested_dir.exists())
        
        result = ensure_directory_exists(nested_dir)
        
        self.assertTrue(nested_dir.exists())
        self.assertEqual(result, nested_dir)

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
        
        self.assertEqual(count, 2)
        self.assertFalse((test_dir / "file1.txt").exists())
        self.assertFalse((test_dir / "file2.txt").exists())
        self.assertTrue((test_dir / "file3.log").exists())

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
        
        self.assertEqual(count, 3)
        self.assertEqual(len(list(test_dir.iterdir())), 0)

    def test_clean_directory_with_nonexistent_directory(self):
        """Test that clean_directory handles non-existent directories."""
        nonexistent = self.temp_dir / "nonexistent"
        count = clean_directory(nonexistent)
        self.assertEqual(count, 0)

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
        
        self.assertEqual(count, 1)
        self.assertFalse((test_dir / "file.txt").exists())
        self.assertTrue(subdir.exists())
        self.assertTrue((subdir / "nested.txt").exists())


if __name__ == '__main__':
    unittest.main()
