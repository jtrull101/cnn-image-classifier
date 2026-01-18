"""Router for analytics endpoints."""

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from img_classifier_api.crud.predictions import get_analytics_summary
from img_classifier_api.database import get_db


class AnalyticsSummaryResponse(BaseModel):
    """Analytics summary response."""

    total_predictions: int
    predictions_by_model: Dict[str, int]
    predictions_by_class: Dict[str, int]
    average_confidence: float
    average_confidence_by_model: Dict[str, float]


class ModelPerformanceResponse(BaseModel):
    """Model performance comparison response."""

    models: Dict[str, Dict[str, Any]]


router = APIRouter()


@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
async def get_summary(
    start_date: Optional[datetime] = Query(None, description="Filter start date"),
    end_date: Optional[datetime] = Query(None, description="Filter end date"),
    db: Session = Depends(get_db),
):
    """
    Get overall analytics summary.

    Provides aggregated statistics including:
    - Total predictions count
    - Predictions grouped by model
    - Predictions grouped by predicted class
    - Average confidence scores overall and by model

    Args:
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        db: Database session

    Returns:
        AnalyticsSummaryResponse: Analytics summary
    """
    summary = get_analytics_summary(db, start_date=start_date, end_date=end_date)

    return AnalyticsSummaryResponse(
        total_predictions=summary["total_predictions"],
        predictions_by_model=summary["predictions_by_model"],
        predictions_by_class=summary["predictions_by_class"],
        average_confidence=summary["average_confidence"],
        average_confidence_by_model=summary["average_confidence_by_model"],
    )


@router.get("/analytics/performance", response_model=ModelPerformanceResponse)
async def get_performance(
    start_date: Optional[datetime] = Query(None, description="Filter start date"),
    end_date: Optional[datetime] = Query(None, description="Filter end date"),
    db: Session = Depends(get_db),
):
    """
    Get detailed model performance comparison.

    Provides per-model statistics for performance analysis and comparison.

    Args:
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        db: Database session

    Returns:
        ModelPerformanceResponse: Performance metrics by model
    """
    from img_classifier_api.models.prediction import PredictionHistory
    from sqlalchemy import func

    query = db.query(PredictionHistory)

    if start_date:
        query = query.filter(PredictionHistory.timestamp >= start_date)

    if end_date:
        query = query.filter(PredictionHistory.timestamp <= end_date)

    # Get performance metrics by model
    models_data: Dict[str, Dict[str, Any]] = {}

    # Group by model and calculate metrics
    model_stats = (
        query.with_entities(
            PredictionHistory.model_name,
            func.count(PredictionHistory.id).label("total_predictions"),
            func.avg(PredictionHistory.confidence).label("avg_confidence"),
            func.min(PredictionHistory.confidence).label("min_confidence"),
            func.max(PredictionHistory.confidence).label("max_confidence"),
        )
        .group_by(PredictionHistory.model_name)
        .all()
    )

    for model_name, total, avg_conf, min_conf, max_conf in model_stats:
        # Get class distribution for this model
        class_distribution = (
            query.filter(PredictionHistory.model_name == model_name)
            .with_entities(
                PredictionHistory.predicted_class,
                func.count(PredictionHistory.id).label("count"),
            )
            .group_by(PredictionHistory.predicted_class)
            .all()
        )

        models_data[model_name] = {
            "total_predictions": total,
            "average_confidence": float(avg_conf) if avg_conf else 0.0,
            "min_confidence": float(min_conf) if min_conf else 0.0,
            "max_confidence": float(max_conf) if max_conf else 0.0,
            "class_distribution": {cls: count for cls, count in class_distribution},
        }

    return ModelPerformanceResponse(models=models_data)
