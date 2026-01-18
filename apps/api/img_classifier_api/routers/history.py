"""Router for prediction history endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from img_classifier_api.crud.predictions import (
    delete_prediction,
    export_predictions,
    get_prediction_by_id,
    get_predictions,
)
from img_classifier_api.database import get_db


class PredictionHistoryItem(BaseModel):
    """Prediction history item for API response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    image_name: str
    image_hash: Optional[str]
    model_name: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    image_thumbnail: Optional[str]
    user_session: Optional[str]


class PredictionHistoryResponse(BaseModel):
    """Response for paginated prediction history."""

    predictions: List[PredictionHistoryItem]
    total: int
    skip: int
    limit: int


class SyncRequest(BaseModel):
    """Request body for syncing local storage with server."""

    predictions: List[Dict[str, Any]] = Field(
        ..., description="List of predictions from local storage"
    )


router = APIRouter()


@router.get("/history", response_model=PredictionHistoryResponse)
async def get_history(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    user_session: Optional[str] = Query(None, description="Filter by user session"),
    db: Session = Depends(get_db),
):
    """
    Get paginated prediction history with optional filters.

    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        model_name: Filter by model name
        user_session: Filter by user session ID
        db: Database session

    Returns:
        PredictionHistoryResponse: Paginated list of predictions
    """
    predictions = get_predictions(
        db, skip=skip, limit=limit, model_name=model_name, user_session=user_session
    )

    # Get total count for pagination
    from img_classifier_api.models.prediction import PredictionHistory as PHModel

    query = db.query(PHModel)
    if model_name:
        query = query.filter(PHModel.model_name == model_name)
    if user_session:
        query = query.filter(PHModel.user_session == user_session)
    total = query.count()

    return PredictionHistoryResponse(
        predictions=[
            PredictionHistoryItem(
                id=cast(int, p.id),
                timestamp=cast(datetime, p.timestamp),
                image_name=cast(str, p.image_name),
                image_hash=cast(Optional[str], p.image_hash),
                model_name=cast(str, p.model_name),
                predicted_class=cast(str, p.predicted_class),
                confidence=cast(float, p.confidence),
                probabilities=p.get_probabilities(),
                image_thumbnail=cast(Optional[str], p.image_thumbnail),
                user_session=cast(Optional[str], p.user_session),
            )
            for p in predictions
        ],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/history/{prediction_id}", response_model=PredictionHistoryItem)
async def get_history_item(
    prediction_id: int,
    db: Session = Depends(get_db),
):
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
        image_hash=cast(Optional[str], prediction.image_hash),
        model_name=cast(str, prediction.model_name),
        predicted_class=cast(str, prediction.predicted_class),
        confidence=cast(float, prediction.confidence),
        probabilities=prediction.get_probabilities(),
        image_thumbnail=cast(Optional[str], prediction.image_thumbnail),
        user_session=cast(Optional[str], prediction.user_session),
    )


@router.delete("/history/{prediction_id}")
async def delete_history_item(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete a prediction by ID.

    Args:
        prediction_id: Prediction ID
        db: Database session

    Returns:
        dict: Success message

    Raises:
        HTTPException: 404 if not found
    """
    success = delete_prediction(db, prediction_id)
    if not success:
        raise HTTPException(404, "Prediction not found")

    return {"message": "Prediction deleted successfully"}


@router.get("/history/export")
async def export_history(
    format: str = Query("json", pattern="^(json|csv)$", description="Export format"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    start_date: Optional[datetime] = Query(None, description="Filter start date"),
    end_date: Optional[datetime] = Query(None, description="Filter end date"),
    db: Session = Depends(get_db),
):
    """
    Export prediction history as CSV or JSON.

    Args:
        format: Export format (json or csv)
        model_name: Filter by model name
        start_date: Filter by start date
        end_date: Filter by end date
        db: Database session

    Returns:
        Response: Exported data with appropriate content type
    """
    exported_data = export_predictions(
        db,
        format=format,
        model_name=model_name,
        start_date=start_date,
        end_date=end_date,
    )

    if format == "csv":
        return Response(
            content=exported_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            },
        )
    else:
        return Response(
            content=exported_data,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            },
        )


@router.post("/history/sync")
async def sync_history(
    request: SyncRequest,
    db: Session = Depends(get_db),
):
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
    from img_classifier_api.crud.predictions import create_prediction
    from img_classifier_api.models.prediction import PredictionHistory

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

        except Exception as e:
            errors.append({"prediction": pred_data.get("image_name", "unknown"), "error": str(e)})

    return {
        "message": "Sync completed",
        "synced": synced,
        "skipped": skipped,
        "errors": errors,
        "total_processed": len(request.predictions),
    }


@router.post("/history/import")
async def import_history(
    # TODO: Implement file upload for importing predictions
    # This would accept a JSON or CSV file and bulk import predictions
    db: Session = Depends(get_db),
):
    """
    Import predictions from a backup file.

    Not yet implemented - placeholder for future functionality.

    Args:
        db: Database session

    Returns:
        dict: Import summary
    """
    raise HTTPException(501, "Import functionality not yet implemented")
