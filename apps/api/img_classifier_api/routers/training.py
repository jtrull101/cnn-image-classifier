"""Router for training endpoints."""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from img_classifier_config import DatasetDetector
from img_classifier_training import TrainingOrchestrator, HyperparameterSpace

logger = logging.getLogger(__name__)


def _training_working_dir() -> Path:
    """Return the working directory for training artefacts.

    Respects ``IMG_CLASSIFIER_WORKING_DIR`` env var; defaults to a user-local
    directory rather than the world-readable ``/tmp``.
    """
    env_dir = os.environ.get("IMG_CLASSIFIER_WORKING_DIR", "").strip()
    if env_dir:
        return Path(env_dir) / "training"
    return Path.home() / ".local" / "share" / "img_classifier" / "training"


class DatasetInfoRequest(BaseModel):
    """Request to get dataset information."""

    dataset_path: str = Field(..., description="Path to dataset directory")


class DatasetInfoResponse(BaseModel):
    """Response with dataset information."""

    dataset_path: str
    has_train_test_split: bool
    num_classes: int
    class_names: List[str]
    total_images: int
    class_distribution: Dict[str, int]
    sample_image_shape: Optional[tuple]
    recommended_complexity: str
    is_balanced: bool


class TrainingJobRequest(BaseModel):
    """Request to start a training job."""

    # Required
    dataset_path: str = Field(..., description="Path to dataset directory")
    project_name: str = Field(..., description="Name for this training project")

    # Basic parameters
    num_epochs: int = Field(25, ge=1, le=200, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, le=256, description="Batch size")
    learning_rate: float = Field(0.001, gt=0, le=1, description="Learning rate")
    architecture_complexity: str = Field(
        "auto", description="Model complexity: auto, simple, medium, deep"
    )

    # Advanced parameters
    dropout_rate: float = Field(0.3, ge=0, lt=1, description="Dropout rate")
    validation_split: float = Field(0.2, gt=0, lt=1, description="Validation split ratio")
    test_split: float = Field(0.5, gt=0, lt=1, description="Test split ratio")
    data_percent: float = Field(1.0, gt=0, le=1, description="Percentage of data to use")

    # Callbacks
    use_early_stopping: bool = Field(True, description="Enable early stopping")
    early_stopping_patience: int = Field(20, ge=1, le=100, description="Early stopping patience")
    use_model_checkpoint: bool = Field(True, description="Save best model checkpoints")
    min_accuracy_to_save: float = Field(0.98, ge=0, le=1, description="Minimum accuracy to save")

    # Hyperparameter optimization
    optimize_hyperparameters: bool = Field(False, description="Enable hyperparameter optimization")
    optimizer_type: str = Field("random", description="Optimizer type: random, grid, bayesian")
    max_trials: int = Field(20, ge=1, le=100, description="Maximum optimization trials")
    search_space_type: str = Field("default", description="Search space: quick, default, full")


class TrainingJobResponse(BaseModel):
    """Response after starting a training job."""

    job_id: str
    status: str
    message: str
    created_at: datetime


class TrainingStatusResponse(BaseModel):
    """Response with training job status."""

    job_id: str
    status: str  # queued, running, completed, failed
    progress: float  # 0-100
    current_epoch: Optional[int]
    total_epochs: Optional[int]
    current_metrics: Optional[Dict[str, float]]
    final_results: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class TrainingJobsListResponse(BaseModel):
    """Response with list of training jobs."""

    jobs: List[TrainingStatusResponse]
    total: int


router = APIRouter()

# In-memory storage for training jobs.
# Known limitation: all job status is lost on server restart.
# Future improvement: persist to the SQLite database used by the predictions router.
training_jobs: Dict[str, Dict[str, Any]] = {}


@router.post("/training/dataset-info", response_model=DatasetInfoResponse)
async def get_dataset_info(request: DatasetInfoRequest) -> DatasetInfoResponse:
    """
    Get information about a dataset.

    Analyzes the dataset structure and returns metadata including
    number of classes, class distribution, recommended architecture, etc.

    Args:
        request: DatasetInfoRequest with dataset path

    Returns:
        DatasetInfoResponse: Dataset information

    Raises:
        HTTPException: 404 if dataset not found, 400 if invalid
    """
    dataset_path = Path(request.dataset_path)

    if not dataset_path.exists():
        raise HTTPException(404, f"Dataset path not found: {request.dataset_path}")

    if not dataset_path.is_dir():
        raise HTTPException(400, f"Path is not a directory: {request.dataset_path}")

    try:
        detector = DatasetDetector(dataset_path)
        info = detector.detect()

        return DatasetInfoResponse(
            dataset_path=str(info["dataset_path"]),
            has_train_test_split=info["has_train_test_split"],
            num_classes=info["num_classes"],
            class_names=info["class_names"],
            total_images=info["total_images"],
            class_distribution=info["class_distribution"],
            sample_image_shape=info.get("sample_image_shape"),
            recommended_complexity=info["recommended_complexity"],
            is_balanced=info["is_balanced"],
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to analyze dataset: {str(e)}")


@router.post("/training/start", response_model=TrainingJobResponse)
async def start_training_job(
    request: TrainingJobRequest, background_tasks: BackgroundTasks
) -> TrainingJobResponse:
    """
    Start a new training job.

    Creates a training job that runs in the background. The job can be
    monitored using the job_id returned in the response.

    Args:
        request: TrainingJobRequest with training parameters
        background_tasks: FastAPI background tasks

    Returns:
        TrainingJobResponse: Job information

    Raises:
        HTTPException: 404 if dataset not found, 400 if invalid parameters
    """
    dataset_path = Path(request.dataset_path)

    if not dataset_path.exists():
        raise HTTPException(404, f"Dataset path not found: {request.dataset_path}")

    # Generate job ID
    job_id = str(uuid.uuid4())
    created_at = datetime.now()

    # Initialize job status
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "current_epoch": None,
        "total_epochs": request.num_epochs,
        "current_metrics": None,
        "final_results": None,
        "error": None,
        "created_at": created_at,
        "started_at": None,
        "completed_at": None,
        "request": request.model_dump(),
    }

    # Start training in background
    background_tasks.add_task(run_training_job, job_id, request)

    return TrainingJobResponse(
        job_id=job_id,
        status="queued",
        message="Training job has been queued and will start shortly",
        created_at=created_at,
    )


@router.get("/training/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str) -> TrainingStatusResponse:
    """
    Get status of a training job.

    Args:
        job_id: Training job ID

    Returns:
        TrainingStatusResponse: Job status and progress

    Raises:
        HTTPException: 404 if job not found
    """
    if job_id not in training_jobs:
        raise HTTPException(404, f"Training job not found: {job_id}")

    job = training_jobs[job_id]

    return TrainingStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        current_epoch=job["current_epoch"],
        total_epochs=job["total_epochs"],
        current_metrics=job["current_metrics"],
        final_results=job["final_results"],
        error=job["error"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
    )


@router.get("/training/jobs", response_model=TrainingJobsListResponse)
async def list_training_jobs(limit: int = 50, skip: int = 0) -> TrainingJobsListResponse:
    """
    List all training jobs.

    Args:
        limit: Maximum number of jobs to return
        skip: Number of jobs to skip

    Returns:
        TrainingJobsListResponse: List of training jobs
    """
    jobs_list = sorted(training_jobs.values(), key=lambda x: x["created_at"], reverse=True)

    paginated_jobs = jobs_list[skip : skip + limit]

    return TrainingJobsListResponse(
        jobs=[
            TrainingStatusResponse(
                job_id=job["job_id"],
                status=job["status"],
                progress=job["progress"],
                current_epoch=job["current_epoch"],
                total_epochs=job["total_epochs"],
                current_metrics=job["current_metrics"],
                final_results=job["final_results"],
                error=job["error"],
                created_at=job["created_at"],
                started_at=job["started_at"],
                completed_at=job["completed_at"],
            )
            for job in paginated_jobs
        ],
        total=len(jobs_list),
    )


@router.delete("/training/jobs/{job_id}")
async def cancel_training_job(job_id: str) -> Dict[str, str]:
    """
    Cancel a training job.

    Note: This only removes the job from the list. Active training
    cannot be stopped once started in the current implementation.

    Args:
        job_id: Training job ID

    Returns:
        dict: Success message

    Raises:
        HTTPException: 404 if job not found
    """
    if job_id not in training_jobs:
        raise HTTPException(404, f"Training job not found: {job_id}")

    del training_jobs[job_id]
    return {"message": "Training job cancelled successfully"}


async def run_training_job(job_id: str, request: TrainingJobRequest) -> None:
    """
    Background task to run training job.

    Args:
        job_id: Training job ID
        request: Training job request parameters
    """
    try:
        # Update status to running
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now()

        dataset_path = Path(request.dataset_path)

        # Create hyperparameter search space if optimization is enabled
        search_space = None
        if request.optimize_hyperparameters:
            if request.search_space_type == "quick":
                search_space = HyperparameterSpace.quick()
            elif request.search_space_type == "default":
                search_space = HyperparameterSpace.default()
            else:
                # Full search space
                search_space = HyperparameterSpace(
                    learning_rates=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                    batch_sizes=[16, 32, 64, 128],
                    dropout_rates=[0.2, 0.3, 0.4, 0.5],
                    architectures=["simple", "medium", "deep"],
                    num_epochs=[20, 30, 40, 50],
                )

        # Detect dataset and create config
        detector = DatasetDetector(dataset_path)
        working_dir = _training_working_dir()
        working_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Training working directory: %s", working_dir)
        config = detector.create_config(
            project_name=request.project_name,
            working_dir=working_dir,
        )

        # Override with user parameters
        config.num_epochs = request.num_epochs
        config.batch_size = request.batch_size
        config.learning_rate = request.learning_rate
        config.dropout_rate = request.dropout_rate
        config.validation_split = request.validation_split
        config.test_split = request.test_split
        config.data_percent = request.data_percent
        config.use_early_stopping = request.use_early_stopping
        config.early_stopping_patience = request.early_stopping_patience
        config.use_model_checkpoint = request.use_model_checkpoint
        config.min_accuracy_to_save = request.min_accuracy_to_save
        config.architecture_complexity = request.architecture_complexity

        # Create orchestrator
        orchestrator = TrainingOrchestrator(
            config=config,
            data_loader=None,
            optimize_hyperparameters=request.optimize_hyperparameters,
            optimizer_type=request.optimizer_type,
            search_space=search_space,
            max_trials=request.max_trials,
        )

        # Run training (simplified - in production, you'd add callbacks for progress)
        results = await asyncio.to_thread(orchestrator.run, plot=False)

        # Update job with results
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100.0
        training_jobs[job_id]["completed_at"] = datetime.now()
        training_jobs[job_id]["final_results"] = {
            "accuracy": float(results["metrics"].get("accuracy", 0)),
            "loss": float(results["metrics"].get("loss", 0)),
            "train_accuracy": float(results["metrics"].get("train_accuracy", 0)),
            "val_accuracy": float(results["metrics"].get("val_accuracy", 0)),
            "test_accuracy": float(results["metrics"].get("test_accuracy", 0)),
            "model_path": str(config.model_path()),
        }

    except Exception as e:
        # Update job with error
        logger.exception("Training job %s failed", job_id)
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["completed_at"] = datetime.now()
