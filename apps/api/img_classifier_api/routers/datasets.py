"""Router for dataset management and download endpoints.

Supports downloading datasets from:
- Any direct URL (GitHub release assets, etc.)
- Kaggle via the Kaggle REST API (username + key authentication)

Downloaded archives (ZIP) are automatically extracted.
"""

import asyncio
import base64
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from img_classifier_api.schemas import MessageResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for download jobs (lost on server restart, same pattern as training jobs)
download_jobs: dict[str, dict[str, Any]] = {}


class DownloadJobStatus(StrEnum):
    """Download job status values."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _datasets_dir() -> Path:
    """Return the directory where datasets are stored.

    Respects ``IMG_CLASSIFIER_WORKING_DIR`` env var; defaults to a user-local
    directory.
    """
    env_dir = os.environ.get("IMG_CLASSIFIER_WORKING_DIR", "").strip()
    if env_dir:
        return Path(env_dir) / "datasets"
    return Path.home() / ".local" / "share" / "img_classifier" / "datasets"


def _safe_dataset_path(datasets_dir: Path, dataset_name: str) -> Path:
    """Resolve dataset path and verify it is inside datasets_dir (path traversal guard)."""
    candidate = (datasets_dir / dataset_name).resolve()
    try:
        candidate.relative_to(datasets_dir.resolve())
    except ValueError as exc:
        raise HTTPException(400, "Invalid dataset name") from exc
    return candidate


def _download_file(url: str, destination: Path, headers: dict[str, str]) -> None:
    """Download a file from *url* to *destination* (run in a thread)."""
    with httpx.Client(follow_redirects=True, timeout=300.0) as client:
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            with open(destination, "wb") as fh:
                for chunk in response.iter_bytes(chunk_size=8192):
                    fh.write(chunk)


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract *zip_path* into *extract_to* (run in a thread)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def _dataset_name_from_url(url: str, override: str | None) -> str:
    """Derive a dataset directory name from the download URL or user override."""
    if override:
        return override
    filename = url.rstrip("/").split("/")[-1]
    # Strip common archive extensions to form a clean folder name
    for ext in (".zip", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"):
        if filename.lower().endswith(ext):
            return filename[: -len(ext)]
    return filename


def _list_datasets_sync(datasets_dir: Path) -> list["DatasetSummary"]:
    """Enumerate datasets directory (blocking I/O, run in a thread)."""
    summaries: list[DatasetSummary] = []
    if datasets_dir.exists():
        for entry in sorted(datasets_dir.iterdir()):
            if entry.is_dir():
                num_items = sum(1 for _ in entry.iterdir())
                summaries.append(
                    DatasetSummary(name=entry.name, path=str(entry), num_items=num_items)
                )
    return summaries


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class DatasetSummary(BaseModel):
    """Summary of a single local dataset."""

    name: str
    path: str
    num_items: int = Field(
        ..., description="Number of top-level items (files + dirs) in the dataset"
    )


class ListDatasetsResponse(BaseModel):
    """Response listing available local datasets."""

    datasets: list[DatasetSummary]
    datasets_dir: str
    total: int


class DownloadUrlRequest(BaseModel):
    """Request to download a dataset from an arbitrary URL."""

    url: str = Field(..., description="Download URL (GitHub release asset, direct link, etc.)")
    name: str | None = Field(
        None, description="Dataset name (defaults to filename without extension)"
    )
    extract: bool = Field(True, description="Extract ZIP archives after download")
    authorization: str | None = Field(
        None, description="Value for the Authorization header (e.g. 'Bearer TOKEN')"
    )


class KaggleDownloadRequest(BaseModel):
    """Request to download a dataset from Kaggle using API credentials."""

    dataset: str = Field(
        ..., description="Kaggle dataset identifier in the form 'owner/dataset-name'"
    )
    kaggle_username: str = Field(..., description="Kaggle username")
    kaggle_key: str = Field(..., description="Kaggle API key")
    name: str | None = Field(None, description="Dataset name (defaults to Kaggle dataset slug)")
    extract: bool = Field(True, description="Extract the downloaded ZIP archive")


class DownloadJobResponse(BaseModel):
    """Response after queuing a download job."""

    job_id: str
    status: DownloadJobStatus
    message: str
    created_at: datetime


class DownloadStatusResponse(BaseModel):
    """Full status of a download job."""

    job_id: str
    status: DownloadJobStatus
    message: str | None
    dataset_name: str | None
    dataset_path: str | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/datasets", response_model=ListDatasetsResponse)
async def list_datasets() -> ListDatasetsResponse:
    """List datasets available in the local datasets directory."""
    datasets_dir = _datasets_dir()
    summaries = await asyncio.to_thread(_list_datasets_sync, datasets_dir)
    return ListDatasetsResponse(
        datasets=summaries,
        datasets_dir=str(datasets_dir),
        total=len(summaries),
    )


@router.post("/datasets/download/url", response_model=DownloadJobResponse)
async def download_dataset_from_url(
    request: DownloadUrlRequest,
    background_tasks: BackgroundTasks,
) -> DownloadJobResponse:
    """Queue a dataset download from any HTTP/HTTPS URL.

    Supports GitHub release assets and any publicly accessible direct link.
    Pass an ``authorization`` value to handle authenticated endpoints
    (e.g. ``'Bearer <token>'``).  ZIP archives are extracted automatically
    when *extract* is ``true``.
    """
    job_id = str(uuid.uuid4())
    created_at = datetime.now()
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": DownloadJobStatus.QUEUED,
        "message": "Download queued",
        "dataset_name": None,
        "dataset_path": None,
        "error": None,
        "created_at": created_at,
        "completed_at": None,
    }
    background_tasks.add_task(_run_url_download, job_id, request)
    return DownloadJobResponse(
        job_id=job_id,
        status=DownloadJobStatus.QUEUED,
        message="Dataset download queued",
        created_at=created_at,
    )


@router.post("/datasets/download/kaggle", response_model=DownloadJobResponse)
async def download_dataset_from_kaggle(
    request: KaggleDownloadRequest,
    background_tasks: BackgroundTasks,
) -> DownloadJobResponse:
    """Queue a dataset download from Kaggle using the Kaggle REST API.

    Authenticates with Basic auth (``kaggle_username:kaggle_key``) against
    ``https://www.kaggle.com/api/v1/datasets/download/<owner>/<dataset>``,
    equivalent to the ``curl`` command recommended in Kaggle's documentation.
    """
    if "/" not in request.dataset:
        raise HTTPException(400, "dataset must be in the format 'owner/dataset-name'")

    job_id = str(uuid.uuid4())
    created_at = datetime.now()
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": DownloadJobStatus.QUEUED,
        "message": "Kaggle download queued",
        "dataset_name": None,
        "dataset_path": None,
        "error": None,
        "created_at": created_at,
        "completed_at": None,
    }
    background_tasks.add_task(_run_kaggle_download, job_id, request)
    return DownloadJobResponse(
        job_id=job_id,
        status=DownloadJobStatus.QUEUED,
        message="Kaggle dataset download queued",
        created_at=created_at,
    )


@router.get("/datasets/download/{job_id}", response_model=DownloadStatusResponse)
async def get_download_status(job_id: str) -> DownloadStatusResponse:
    """Get the status of a dataset download job."""
    if job_id not in download_jobs:
        raise HTTPException(404, f"Download job not found: {job_id}")

    job = download_jobs[job_id]
    return DownloadStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        message=job["message"],
        dataset_name=job["dataset_name"],
        dataset_path=job["dataset_path"],
        error=job["error"],
        created_at=job["created_at"],
        completed_at=job["completed_at"],
    )


@router.delete("/datasets/{dataset_name}", response_model=MessageResponse)
async def delete_dataset(dataset_name: str) -> MessageResponse:
    """Delete a local dataset directory."""
    datasets_dir = _datasets_dir()
    dataset_path = _safe_dataset_path(datasets_dir, dataset_name)

    if not dataset_path.exists():
        raise HTTPException(404, f"Dataset not found: {dataset_name}")
    if not dataset_path.is_dir():
        raise HTTPException(400, f"Not a dataset directory: {dataset_name}")

    await asyncio.to_thread(shutil.rmtree, dataset_path)
    return MessageResponse(message=f"Dataset '{dataset_name}' deleted successfully")


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------


async def _run_url_download(job_id: str, request: DownloadUrlRequest) -> None:
    """Background task: download a dataset from an arbitrary URL."""
    try:
        download_jobs[job_id]["status"] = DownloadJobStatus.RUNNING
        download_jobs[job_id]["message"] = "Downloading…"

        datasets_dir = _datasets_dir()
        datasets_dir.mkdir(parents=True, exist_ok=True)

        headers: dict[str, str] = {}
        if request.authorization:
            headers["Authorization"] = request.authorization

        url_filename = request.url.rstrip("/").split("/")[-1]
        dataset_name = _dataset_name_from_url(request.url, request.name)
        download_path = datasets_dir / url_filename

        await asyncio.to_thread(_download_file, request.url, download_path, headers)

        if request.extract and zipfile.is_zipfile(download_path):
            extract_dir = datasets_dir / dataset_name
            extract_dir.mkdir(exist_ok=True)
            await asyncio.to_thread(_extract_zip, download_path, extract_dir)
            download_path.unlink()
            final_path = extract_dir
        else:
            final_path = download_path

        download_jobs[job_id].update(
            {
                "status": DownloadJobStatus.COMPLETED,
                "message": "Download completed successfully",
                "dataset_name": dataset_name,
                "dataset_path": str(final_path),
                "completed_at": datetime.now(),
            }
        )

    except Exception as exc:
        logger.exception("URL download job %s failed", job_id)
        download_jobs[job_id].update(
            {
                "status": DownloadJobStatus.FAILED,
                "message": None,
                "error": str(exc),
                "completed_at": datetime.now(),
            }
        )


async def _run_kaggle_download(job_id: str, request: KaggleDownloadRequest) -> None:
    """Background task: download a dataset from Kaggle."""
    try:
        download_jobs[job_id]["status"] = DownloadJobStatus.RUNNING
        download_jobs[job_id]["message"] = "Downloading from Kaggle…"

        datasets_dir = _datasets_dir()
        datasets_dir.mkdir(parents=True, exist_ok=True)

        owner, slug = request.dataset.split("/", 1)
        dataset_name = request.name or slug
        url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}"

        credentials = f"{request.kaggle_username}:{request.kaggle_key}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers = {"Authorization": f"Basic {encoded}"}

        zip_filename = f"{slug}.zip"
        download_path = datasets_dir / zip_filename

        await asyncio.to_thread(_download_file, url, download_path, headers)

        if request.extract and zipfile.is_zipfile(download_path):
            extract_dir = datasets_dir / dataset_name
            extract_dir.mkdir(exist_ok=True)
            await asyncio.to_thread(_extract_zip, download_path, extract_dir)
            download_path.unlink()
            final_path = extract_dir
        else:
            final_path = download_path

        download_jobs[job_id].update(
            {
                "status": DownloadJobStatus.COMPLETED,
                "message": "Kaggle download completed successfully",
                "dataset_name": dataset_name,
                "dataset_path": str(final_path),
                "completed_at": datetime.now(),
            }
        )

    except Exception as exc:
        logger.exception("Kaggle download job %s failed", job_id)
        download_jobs[job_id].update(
            {
                "status": DownloadJobStatus.FAILED,
                "message": None,
                "error": str(exc),
                "completed_at": datetime.now(),
            }
        )
