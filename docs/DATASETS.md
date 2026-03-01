# Dataset Management Guide

This guide explains how datasets are managed in this project and how to download them for training or experimentation.

## Overview

To keep the Git repository lightweight and fast to clone, datasets are **not stored in the repository**. Instead, they are hosted on GitHub Releases and can be downloaded on-demand when needed.

**Total Size:** ~461 MB (all datasets combined)

## Why Not Include Datasets in Git?

- **Repository Size:** Large binary files bloat the repository, making clones slow
- **Git Performance:** Git is optimized for text files, not large binary datasets
- **Flexibility:** Users only download what they need
- **Version Control:** Datasets can be versioned independently via releases
- **CI/CD Efficiency:** Continuous integration can skip dataset downloads

## Available Datasets

### 1. Alzheimer's MRI Dataset
- **File:** `alzheimers-mri-dataset.zip`
- **Size:** 72 MB
- **Description:** MRI brain scans classified into 4 Alzheimer's severity levels
- **Categories:**
  - No Impairment
  - Very Mild Impairment
  - Mild Impairment
  - Moderate Impairment
- **Source:** [Kaggle - Best Alzheimer MRI Dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy)
- **Use Case:** Primary dataset for this project's Alzheimer's classification system

### 2. Brain Tumor MRI Dataset
- **File:** `brain-tumor-mri-dataset.zip`
- **Size:** 149 MB
- **Description:** MRI scans for brain tumor classification
- **Use Case:** Additional medical imaging dataset for testing generalized system

### 3. COVID Chest X-Ray Dataset
- **File:** `covid-chest-xray.zip`
- **Size:** 241 MB
- **Description:** Chest X-ray images for COVID-19 classification
- **Use Case:** Demonstrates system versatility across imaging modalities

## Quick Start

### Download All Datasets

```bash
# Using Make (recommended)
make download-datasets

# Or directly with Python
uv run python scripts/download_datasets.py
```

### Download Specific Dataset

```bash
# Using Python script
uv run python scripts/download_datasets.py alzheimers-mri-dataset
uv run python scripts/download_datasets.py brain-tumor-mri-dataset
uv run python scripts/download_datasets.py covid-chest-xray
```

### List Available Datasets

```bash
# Using Make
make list-datasets

# Or with Python
uv run python scripts/download_datasets.py --list
```

**Example output:**
```
=== Available Datasets ===

Name: alzheimers-mri-dataset
Size: 72 MB
Status: ✓ Downloaded
Description: Alzheimer's MRI classification dataset with 4 severity levels

Name: brain-tumor-mri-dataset
Size: 149 MB
Status: ✗ Not downloaded
Description: Brain tumor MRI dataset for classification tasks

Total size (all datasets): 462 MB
```

## Download Script Features

The download utility (`scripts/download_datasets.py`) provides:

✅ **Skip Existing Files** - Won't re-download if file already exists  
✅ **Progress Bar** - Visual feedback during download using `tqdm`  
✅ **Error Handling** - Graceful failures with cleanup of partial downloads  
✅ **Selective Downloads** - Download all or specific datasets  
✅ **Status Check** - See which datasets are already downloaded  

## When Do You Need Datasets?

### ✅ You NEED datasets if you want to:
- Train new models from scratch
- Fine-tune existing models
- Experiment with different architectures
- Run data augmentation experiments
- Perform hyperparameter optimization
- Contribute model improvements

### ❌ You DON'T NEED datasets if you:
- Only want to use the pre-trained models
- Are running the API for inference
- Are just exploring the codebase
- Are running tests (tests use mock data)
- Are working on non-training code (utils, API, etc.)

## CI/CD Integration

### GitHub Actions

Datasets can be downloaded in CI pipelines when needed:

```yaml
- name: Download datasets for training tests
  run: make download-datasets
  if: matrix.test-type == 'training'
```

### Selective Testing

Most tests use mock data and don't require real datasets:

```bash
# Unit tests (fast, no datasets needed)
make test-unit

# Integration tests (may require datasets)
make test-integration
```

## Manual Download (Fallback)

If the automated download fails, you can manually download datasets:

1. Visit the [GitHub Releases page](https://github.com/jtrull101/alz-mri-neural-network/releases)
2. Find the release tagged `v1.0.0-datasets`
3. Download the desired `.zip` files
4. Place them in the `datasets/` directory at the repository root

## Dataset Structure

Once downloaded and extracted, datasets follow this structure:

```
datasets/
├── manifest.json                          # Metadata and download URLs
├── alzheimers-mri-dataset.zip            # Downloaded (if you ran download)
├── brain-tumor-mri-dataset.zip           # Downloaded (if you ran download)
├── covid-chest-xray.zip                  # Downloaded (if you ran download)
└── [extracted folders appear here after training runs]
```

**Note:** The `.zip` files are in `.gitignore` and will never be committed to the repository.

## Updating Dataset URLs

If you're a maintainer and need to update dataset URLs:

1. Edit `datasets/manifest.json`
2. Update the `url` field for each dataset
3. Optionally update `size_mb` and `description`
4. Commit the manifest changes

## Troubleshooting

### Download Fails

**Problem:** Download times out or fails mid-transfer

**Solution:**
```bash
# Try again - the script will resume if file is partial
uv run python scripts/download_datasets.py <dataset-name>

# Or manually download from GitHub Releases
```

### Missing Dependencies

**Problem:** `ModuleNotFoundError: No module named 'requests'` or `'tqdm'`

**Solution:**
```bash
# Sync all dependencies
uv sync --all-groups

# Or install specific packages
uv pip install requests tqdm
```

### Wrong Download Location

**Problem:** Script downloads to wrong directory

**Solution:**
```bash
# Always run from repository root
cd /path/to/alz-mri-neural-network
make download-datasets
```

### Datasets Not Recognized

**Problem:** Training scripts can't find datasets after download

**Solution:**
- Ensure `.zip` files are in `datasets/` directory
- Some training scripts may need explicit path configuration
- Check if extraction is required (some scripts auto-extract)

## Storage Recommendations

### Local Development
- Download only the datasets you're actively using
- Delete extracted folders when not needed (keep `.zip` files)
- Use `.gitignore` patterns to prevent accidental commits

### Shared Environments
- Download once to a shared location
- Symlink to `datasets/` from multiple clones
- Use network storage for team access

### Cloud Environments
- Download during container build or initialization
- Cache in CI/CD pipeline for faster builds
- Use cloud storage mounts for large datasets

## Related Documentation

- [README.md](../README.md) - Quick start guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [GENERALIZED_SYSTEM.md](GENERALIZED_SYSTEM.md) - Training with any dataset
- [TESTING.md](TESTING.md) - Testing guide

## Questions?

If you encounter issues with dataset downloads:

1. Check the [GitHub Issues](https://github.com/jtrull101/alz-mri-neural-network/issues)
2. Verify your internet connection and GitHub access
3. Try manual download from Releases as a fallback
4. Open a new issue if the problem persists

---

**Note:** Datasets are for educational and research purposes only. The models trained on these datasets should not be used for medical diagnosis.
