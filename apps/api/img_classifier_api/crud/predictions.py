"""CRUD operations for prediction history."""

import csv
import io
import json
from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from img_classifier_api.models.prediction import PredictionHistory
from img_classifier_api.schemas import ExportFormat


def create_prediction(db: Session, prediction_data: dict[str, Any]) -> PredictionHistory:
    """Create a new prediction record in the database.

    Args:
        db: Database session
        prediction_data: Dictionary containing prediction data with keys:
            - image_name: str
            - image_hash: str (optional)
            - model_name: str
            - predicted_class: str
            - confidence: float
            - probabilities: dict[str, float]
            - image_thumbnail: str (optional, base64 encoded)
            - user_session: str (optional)

    Returns:
        PredictionHistory: Created prediction record
    """
    db_prediction = PredictionHistory(
        image_name=prediction_data["image_name"],
        image_hash=prediction_data.get("image_hash"),
        model_name=prediction_data["model_name"],
        predicted_class=prediction_data["predicted_class"],
        confidence=prediction_data["confidence"],
        probabilities=json.dumps(prediction_data["probabilities"]),
        image_thumbnail=prediction_data.get("image_thumbnail"),
        user_session=prediction_data.get("user_session"),
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction


def get_predictions(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    model_name: str | None = None,
    user_session: str | None = None,
) -> list[PredictionHistory]:
    """Get paginated list of predictions with optional filters.

    Args:
        db: Database session
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        model_name: Filter by model name (optional)
        user_session: Filter by user session (optional)

    Returns:
        list[PredictionHistory]: List of prediction records
    """
    query = db.query(PredictionHistory)

    if model_name:
        query = query.filter(PredictionHistory.model_name == model_name)

    if user_session:
        query = query.filter(PredictionHistory.user_session == user_session)

    return query.order_by(PredictionHistory.timestamp.desc()).offset(skip).limit(limit).all()


def get_prediction_by_id(db: Session, prediction_id: int) -> PredictionHistory | None:
    """Get a single prediction by ID.

    Args:
        db: Database session
        prediction_id: Prediction ID

    Returns:
        PredictionHistory | None: Prediction record or None if not found
    """
    return db.query(PredictionHistory).filter(PredictionHistory.id == prediction_id).first()


def delete_prediction(db: Session, prediction_id: int) -> bool:
    """Delete a prediction by ID.

    Args:
        db: Database session
        prediction_id: Prediction ID

    Returns:
        bool: True if deleted, False if not found
    """
    db_prediction = get_prediction_by_id(db, prediction_id)
    if db_prediction:
        db.delete(db_prediction)
        db.commit()
        return True
    return False


def count_predictions(
    db: Session,
    model_name: str | None = None,
    user_session: str | None = None,
) -> int:
    """Count predictions with optional filters.

    Args:
        db: Database session
        model_name: Filter by model name (optional)
        user_session: Filter by user session (optional)

    Returns:
        int: Number of matching prediction records
    """
    from sqlalchemy import func

    query = db.query(func.count(PredictionHistory.id))
    if model_name:
        query = query.filter(PredictionHistory.model_name == model_name)
    if user_session:
        query = query.filter(PredictionHistory.user_session == user_session)
    return query.scalar() or 0


def get_predictions_by_model(
    db: Session,
    model_name: str,
    skip: int = 0,
    limit: int = 100,
) -> list[PredictionHistory]:
    """Get all predictions for a specific model.

    Args:
        db: Database session
        model_name: Model name to filter by
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        list[PredictionHistory]: List of prediction records
    """
    return (
        db.query(PredictionHistory)
        .filter(PredictionHistory.model_name == model_name)
        .order_by(PredictionHistory.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def export_predictions(
    db: Session,
    export_format: ExportFormat = ExportFormat.JSON,
    model_name: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> str:
    """Export predictions as CSV or JSON.

    Args:
        db: Database session
        format: Export format ("json" or "csv")
        model_name: Filter by model name (optional)
        start_date: Filter by start date (optional)
        end_date: Filter by end date (optional)

    Returns:
        str: Exported data as string (CSV or JSON)
    """
    query = db.query(PredictionHistory)

    if model_name:
        query = query.filter(PredictionHistory.model_name == model_name)

    if start_date:
        query = query.filter(PredictionHistory.timestamp >= start_date)

    if end_date:
        query = query.filter(PredictionHistory.timestamp <= end_date)

    predictions = query.order_by(PredictionHistory.timestamp.desc()).all()

    if export_format == ExportFormat.CSV:
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "id",
                "timestamp",
                "image_name",
                "model_name",
                "predicted_class",
                "confidence",
            ],
        )
        writer.writeheader()
        for pred in predictions:
            writer.writerow(
                {
                    "id": pred.id,
                    "timestamp": pred.timestamp.isoformat() if pred.timestamp is not None else "",
                    "image_name": pred.image_name,
                    "model_name": pred.model_name,
                    "predicted_class": pred.predicted_class,
                    "confidence": pred.confidence,
                }
            )
        return output.getvalue()
    else:  # ExportFormat.JSON
        return json.dumps([pred.to_dict() for pred in predictions], indent=2)


def get_analytics_summary(
    db: Session,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, Any]:
    """Get analytics summary for the dashboard.

    Args:
        db: Database session
        start_date: Start date for filtering (optional)
        end_date: End date for filtering (optional)

    Returns:
        dict: Analytics summary with stats
    """
    query = db.query(PredictionHistory)

    if start_date:
        query = query.filter(PredictionHistory.timestamp >= start_date)

    if end_date:
        query = query.filter(PredictionHistory.timestamp <= end_date)

    # Total predictions
    total_predictions = query.count()

    # Predictions by model
    predictions_by_model = (
        query.with_entities(
            PredictionHistory.model_name,
            func.count(PredictionHistory.id).label("count"),
        )
        .group_by(PredictionHistory.model_name)
        .all()
    )

    # Predictions by class
    predictions_by_class = (
        query.with_entities(
            PredictionHistory.predicted_class,
            func.count(PredictionHistory.id).label("count"),
        )
        .group_by(PredictionHistory.predicted_class)
        .all()
    )

    # Average confidence
    avg_confidence = query.with_entities(func.avg(PredictionHistory.confidence)).scalar() or 0.0

    # Average confidence by model
    avg_confidence_by_model = (
        query.with_entities(
            PredictionHistory.model_name,
            func.avg(PredictionHistory.confidence).label("avg_confidence"),
        )
        .group_by(PredictionHistory.model_name)
        .all()
    )

    return {
        "total_predictions": total_predictions,
        "predictions_by_model": {model: count for model, count in predictions_by_model},
        "predictions_by_class": {cls: count for cls, count in predictions_by_class},
        "average_confidence": float(avg_confidence),
        "average_confidence_by_model": {
            model: float(conf) for model, conf in avg_confidence_by_model
        },
    }
