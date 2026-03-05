"""Router for prediction history endpoints."""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from img_classifier_api.crud.predictions import (
    create_prediction,
    delete_prediction,
    export_predictions,
    get_prediction_by_id,
    get_predictions,
)
from img_classifier_api.database import get_db
from img_classifier_api.models.prediction import PredictionHistory
from img_classifier_api.routers._constants import (
    _HTMX_REQUEST_HEADER,
    _HTMX_REQUEST_VALUE,
    _MEDIA_TYPE_JSON,
    _MEDIA_TYPE_CSV,
)
from img_classifier_api.schemas import MessageResponse

_TIMESTAMP_FORMAT = "%b %d, %Y %I:%M %p"
_FILENAME_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


class ExportFormat(str, Enum):
    """Supported export formats for prediction history."""

    JSON = "json"
    CSV = "csv"


class PredictionHistoryItem(BaseModel):
    """Prediction history item for API response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    image_name: str
    image_hash: str | None
    model_name: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    image_thumbnail: str | None
    user_session: str | None


class PredictionHistoryResponse(BaseModel):
    """Response for paginated prediction history."""

    predictions: list[PredictionHistoryItem]
    total: int
    skip: int
    limit: int


class SyncRequest(BaseModel):
    """Request body for syncing local storage with server."""

    predictions: list[dict[str, Any]] = Field(
        ..., description="List of predictions from local storage"
    )


class SyncError(BaseModel):
    """A single sync failure entry."""

    prediction: str
    error: str


class SyncResponse(BaseModel):
    """Response for the history/sync endpoint."""

    message: str
    synced: int
    skipped: int
    errors: list[SyncError]
    total_processed: int


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/history", response_model=None)
async def get_history(
    request: Request,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    model_name: str | None = Query(None, description="Filter by model name"),
    user_session: str | None = Query(None, description="Filter by user session"),
    db: Session = Depends(get_db),
) -> HTMLResponse | PredictionHistoryResponse:
    """
    Get paginated prediction history with optional filters.

    Returns HTML for HTMX requests, JSON for API requests.

    Args:
        request: FastAPI request object
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        model_name: Filter by model name
        user_session: Filter by user session ID
        db: Database session

    Returns:
        HTMLResponse or PredictionHistoryResponse: History data
    """
    predictions = get_predictions(
        db, skip=skip, limit=limit, model_name=model_name, user_session=user_session
    )

    # Get total count for pagination
    query = db.query(PredictionHistory)
    if model_name:
        query = query.filter(PredictionHistory.model_name == model_name)
    if user_session:
        query = query.filter(PredictionHistory.user_session == user_session)
    total = query.count()

    # Check if this is an HTMX request
    is_htmx = request.headers.get(_HTMX_REQUEST_HEADER) == _HTMX_REQUEST_VALUE

    if is_htmx:
        # Return HTML for HTMX
        if not predictions:
            return HTMLResponse(
                """
                <div class="text-center text-gray-500 py-4">
                    <p>No predictions yet</p>
                </div>
                """
            )

        html_parts = []
        for p in predictions:
            timestamp_str = (
                p.timestamp.strftime(_TIMESTAMP_FORMAT) if p.timestamp is not None else "Unknown"
            )
            confidence_pct = f"{p.confidence * 100:.1f}%" if p.confidence is not None else "N/A"

            html_parts.append(
                f"""
                <div class="flex items-center justify-between p-3 border-b border-gray-200 hover:bg-gray-50 transition">
                    <div class="flex-1">
                        <div class="flex items-center gap-2 mb-1">
                            <p class="font-semibold text-gray-900">{p.predicted_class}</p>
                            <span class="badge badge-sm badge-success">{confidence_pct}</span>
                        </div>
                        <p class="text-xs text-gray-500">{p.image_name}</p>
                        <p class="text-xs text-gray-400">{timestamp_str}</p>
                    </div>
                    <div class="text-xs text-gray-600">
                        {p.model_name}
                    </div>
                </div>
                """
            )

        return HTMLResponse("".join(html_parts))

    # Return JSON for API requests
    return PredictionHistoryResponse(
        predictions=[
            PredictionHistoryItem(
                id=cast(int, p.id),
                timestamp=cast(datetime, p.timestamp),
                image_name=cast(str, p.image_name),
                image_hash=cast(str | None, p.image_hash),
                model_name=cast(str, p.model_name),
                predicted_class=cast(str, p.predicted_class),
                confidence=cast(float, p.confidence),
                probabilities=p.get_probabilities(),
                image_thumbnail=cast(str | None, p.image_thumbnail),
                user_session=cast(str | None, p.user_session),
            )
            for p in predictions
        ],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/history/export")
async def export_history(
    export_format: ExportFormat = Query(ExportFormat.JSON, description="Export format"),
    model_name: str | None = Query(None, description="Filter by model name"),
    start_date: datetime | None = Query(None, description="Filter start date"),
    end_date: datetime | None = Query(None, description="Filter end date"),
    db: Session = Depends(get_db),
) -> Response:
    """
    Export prediction history as CSV or JSON.

    Args:
        export_format: Export format (json or csv)
        model_name: Filter by model name
        start_date: Filter by start date
        end_date: Filter by end date
        db: Database session

    Returns:
        Response: Exported data with appropriate content type
    """
    exported_data = export_predictions(
        db,
        format=export_format.value,
        model_name=model_name,
        start_date=start_date,
        end_date=end_date,
    )

    ts = datetime.now().strftime(_FILENAME_TIMESTAMP_FORMAT)
    if export_format is ExportFormat.CSV:
        return Response(
            content=exported_data,
            media_type=_MEDIA_TYPE_CSV,
            headers={"Content-Disposition": f"attachment; filename=predictions_{ts}.csv"},
        )
    return Response(
        content=exported_data,
        media_type=_MEDIA_TYPE_JSON,
        headers={"Content-Disposition": f"attachment; filename=predictions_{ts}.json"},
    )


@router.get("/history/{prediction_id}", response_model=PredictionHistoryItem)
async def get_history_item(
    prediction_id: int,
    db: Session = Depends(get_db),
) -> PredictionHistoryItem:
    """
    Get a single prediction by ID.

    Args:
        prediction_id: Prediction ID
        db: Database session

    Returns:
        PredictionHistoryItem: Prediction details

    Raises:
        HTTPException: 404 if not found
    """
    prediction = get_prediction_by_id(db, prediction_id)
    if not prediction:
        raise HTTPException(404, "Prediction not found")

    return PredictionHistoryItem(
        id=cast(int, prediction.id),
        timestamp=cast(datetime, prediction.timestamp),
        image_name=cast(str, prediction.image_name),
        image_hash=cast(str | None, prediction.image_hash),
        model_name=cast(str, prediction.model_name),
        predicted_class=cast(str, prediction.predicted_class),
        confidence=cast(float, prediction.confidence),
        probabilities=prediction.get_probabilities(),
        image_thumbnail=cast(str | None, prediction.image_thumbnail),
        user_session=cast(str | None, prediction.user_session),
    )


@router.delete("/history/{prediction_id}", response_model=MessageResponse)
async def delete_history_item(
    prediction_id: int,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """
    Delete a prediction by ID.

    Args:
        prediction_id: Prediction ID
        db: Database session

    Returns:
        MessageResponse: Success message

    Raises:
        HTTPException: 404 if not found
    """
    success = delete_prediction(db, prediction_id)
    if not success:
        raise HTTPException(404, "Prediction not found")

    return MessageResponse(message="Prediction deleted successfully")


@router.post("/history/sync", response_model=SyncResponse)
async def sync_history(
    request: SyncRequest,
    db: Session = Depends(get_db),
) -> SyncResponse:
    """
    Sync local storage predictions with server database.

    This endpoint accepts predictions from browser localStorage and
    saves them to the server database, avoiding duplicates based on
    image_hash and timestamp.

    Args:
        request: SyncRequest with list of predictions
        db: Database session

    Returns:
        dict: Summary of sync operation
    """
    synced = 0
    skipped = 0
    errors = []

    for pred_data in request.predictions:
        try:
            # Check if prediction already exists (by hash)
            image_hash = pred_data.get("image_hash")

            if image_hash:
                existing = (
                    db.query(PredictionHistory)
                    .filter(PredictionHistory.image_hash == image_hash)
                    .first()
                )

                if existing:
                    skipped += 1
                    continue

            # Create new prediction
            create_prediction(db, pred_data)
            synced += 1

        except (ValueError, SQLAlchemyError) as e:
            logger.exception(
                "Failed to sync prediction '%s'", pred_data.get("image_name", "unknown")
            )
            errors.append({"prediction": pred_data.get("image_name", "unknown"), "error": str(e)})

    return SyncResponse(
        message="Sync completed",
        synced=synced,
        skipped=skipped,
        errors=[SyncError(**e) for e in errors],
        total_processed=len(request.predictions),
    )
