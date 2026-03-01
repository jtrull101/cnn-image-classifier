#!/usr/bin/env python3
"""
Dataset Download Utility

Downloads datasets from GitHub Releases to avoid bloating the Git repository.
Supports downloading all datasets or specific ones by name.

Usage:
    python scripts/download_datasets.py                    # Download all datasets
    python scripts/download_datasets.py alzheimers-mri-dataset  # Download specific dataset
    python scripts/download_datasets.py --list             # List available datasets
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Error: Required dependencies not found.")
    print("Please install them with: uv pip install requests tqdm")
    sys.exit(1)


class DatasetDownloader:
    """Handles downloading datasets from GitHub Releases."""

    def __init__(self, manifest_path: Path) -> None:
        """Initialize downloader with manifest file."""
        self.manifest_path = manifest_path
        self.datasets_dir = manifest_path.parent
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        """Load dataset manifest from JSON file."""
        try:
            with self.manifest_path.open() as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Manifest file not found at {self.manifest_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in manifest file: {e}")
            sys.exit(1)

    def list_datasets(self) -> None:
        """List all available datasets with their info."""
        print("\n=== Available Datasets ===\n")
        total_size = 0

        for dataset in self.manifest["datasets"]:
            name = dataset["name"]
            size = dataset["size_mb"]
            desc = dataset["description"]
            filepath = self.datasets_dir / dataset["filename"]
            exists = filepath.exists()
            status = "[Downloaded]" if exists else "[Not downloaded]"

            print(f"Name: {name}")
            print(f"Size: {size} MB")
            print(f"Status: {status}")
            print(f"Description: {desc}")
            print()

            total_size += size

        print(f"Total size (all datasets): {total_size} MB")
        print()

    def download_dataset(self, dataset_name: str) -> bool:
        """Download a specific dataset by name."""
        dataset = self._find_dataset(dataset_name)
        if not dataset:
            print(f"Error: Dataset '{dataset_name}' not found in manifest.")
            print("Run with --list to see available datasets.")
            return False

        filename = dataset["filename"]
        filepath = self.datasets_dir / filename
        url = dataset["url"]

        # Check if file already exists
        if filepath.exists():
            print(f"[SKIP] {filename} already exists (skipping download)")
            return True

        print(f"Downloading {filename} ({dataset['size_mb']} MB)...")
        print(f"From: {url}")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Get total file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with (
                filepath.open("wb") as f,
                tqdm(
                    desc=filename,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"[OK] Successfully downloaded {filename}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error downloading {filename}: {e}")
            # Clean up partial download
            if filepath.exists():
                filepath.unlink()
            return False

    def download_all(self) -> bool:
        """Download all datasets listed in manifest."""
        print(f"\n=== Downloading {len(self.manifest['datasets'])} datasets ===\n")

        success_count = 0
        for dataset in self.manifest["datasets"]:
            if self.download_dataset(dataset["name"]):
                success_count += 1
            print()  # Blank line between downloads

        total = len(self.manifest["datasets"])
        print(f"\nDownload complete: {success_count}/{total} successful")
        return success_count == total

    def _find_dataset(self, name: str) -> dict[str, Any] | None:
        """Find dataset by name in manifest."""
        for dataset in self.manifest["datasets"]:
            if dataset["name"] == name:
                return dataset
        return None


def main() -> None:
    """Main entry point for dataset downloader."""
    parser = argparse.ArgumentParser(
        description="Download datasets for the image classification project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_datasets.py                         # Download all
  python scripts/download_datasets.py alzheimers-mri-dataset  # Download one
  python scripts/download_datasets.py --list                  # List datasets
        """,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Name of specific dataset to download (omit to download all)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/manifest.json"),
        help="Path to manifest file (default: datasets/manifest.json)",
    )

    args = parser.parse_args()

    # Resolve manifest path relative to repository root
    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = repo_root / args.manifest

    downloader = DatasetDownloader(manifest_path)

    if args.list:
        downloader.list_datasets()
        return

    if args.dataset:
        # Download specific dataset
        success = downloader.download_dataset(args.dataset)
        sys.exit(0 if success else 1)
    else:
        # Download all datasets
        success = downloader.download_all()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
