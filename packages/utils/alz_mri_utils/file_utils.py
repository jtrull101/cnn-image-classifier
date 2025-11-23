"""Utility functions for file operations."""

import shutil
import zipfile
from pathlib import Path
import gdown
from pyunpack import Archive


def download_from_google_drive(file_id: str, destination: Path, quiet: bool = False) -> bool:
    """Download a file from Google Drive.

    Args:
        file_id: Google Drive file ID
        destination: Path where file should be saved
        quiet: Whether to suppress output

    Returns:
        True if successful, False otherwise
    """
    try:
        url = f"https://docs.google.com/uc?id={file_id}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(destination), quiet=quiet)
        return True
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract an archive file.

    Args:
        archive_path: Path to the archive file
        extract_to: Directory to extract to

    Returns:
        True if successful, False otherwise
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            Archive(str(archive_path)).extractall(str(extract_to))

        return True
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return False


def organize_dataset(source_dir: Path, dest_dir: Path) -> bool:
    """Organize dataset from nested structure to flat structure.

    Moves files from source_dir/category/subcategory/file.jpg
    to dest_dir/category/file.jpg

    Args:
        source_dir: Source directory with nested structure
        dest_dir: Destination directory for flat structure

    Returns:
        True if successful, False otherwise
    """
    try:
        if not source_dir.exists():
            return False

        for split_dir in source_dir.iterdir():
            if not split_dir.is_dir():
                continue

            for category in split_dir.iterdir():
                if not category.is_dir():
                    continue

                target_dir = dest_dir / split_dir.name / category.name
                target_dir.mkdir(parents=True, exist_ok=True)

                for file_path in category.iterdir():
                    if file_path.is_file():
                        dest_file = target_dir / file_path.name
                        if not dest_file.exists():
                            shutil.copy2(file_path, dest_file)

        return True
    except Exception as e:
        print(f"Error organizing dataset: {e}")
        return False


def clean_directory(directory: Path, pattern: str = "*") -> int:
    """Remove files matching pattern from directory.

    Args:
        directory: Directory to clean
        pattern: File pattern to match (default: all files)

    Returns:
        Number of files removed
    """
    count = 0
    if directory.exists():
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    count += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    return count


def ensure_directory_exists(directory: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path

    Returns:
        The directory path
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory
