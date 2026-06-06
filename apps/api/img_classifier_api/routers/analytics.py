"""Router for analytics endpoints."""

import html
from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from img_classifier_api.crud.predictions import get_analytics_summary
from img_classifier_api.database import get_db
from img_classifier_api.models.prediction import PredictionHistory
from img_classifier_api.routers._constants import _HTMX_REQUEST_HEADER, _HTMX_REQUEST_VALUE


class AnalyticsSummaryResponse(BaseModel):
    """Analytics summary response."""

    total_predictions: int
    predictions_by_model: dict[str, int]
    predictions_by_class: dict[str, int]
    average_confidence: float
    average_confidence_by_model: dict[str, float]


class ModelPerformanceResponse(BaseModel):
    """Model performance comparison response."""

    models: dict[str, dict[str, Any]]


_SUMMARY_TOP_N = 3  # Maximum rows shown per section in the HTMX summary widget

router = APIRouter()


@router.get("/analytics/summary", response_model=None)
async def get_summary(
    request: Request,
    start_date: datetime | None = Query(None, description="Filter start date"),
    end_date: datetime | None = Query(None, description="Filter end date"),
    db: Session = Depends(get_db),
) -> HTMLResponse | AnalyticsSummaryResponse:
    """Get overall analytics summary.

    Returns HTML for HTMX requests, JSON for API requests.

    Provides aggregated statistics including:
    - Total predictions count
    - Predictions grouped by model
    - Predictions grouped by predicted class
    - Average confidence scores overall and by model

    Args:
        request: FastAPI request object
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        db: Database session

    Returns:
        HTMLResponse or AnalyticsSummaryResponse: Analytics summary
    """
    summary = get_analytics_summary(db, start_date=start_date, end_date=end_date)

    # Check if this is an HTMX request
    is_htmx = request.headers.get(_HTMX_REQUEST_HEADER) == _HTMX_REQUEST_VALUE

    if is_htmx:
        # Return HTML for HTMX
        predictions_by_model = summary.get("predictions_by_model", {})
        predictions_by_class = summary.get("predictions_by_class", {})

        model_rows = []
        for model, count in list(predictions_by_model.items())[:_SUMMARY_TOP_N]:
            model_rows.append(
                f"""
                <div class="flex justify-between text-sm py-1">
                    <span class="text-gray-600">{html.escape(model)}</span>
                    <span class="font-semibold">{count}</span>
                </div>
                """
            )

        class_rows = []
        for cls, count in list(predictions_by_class.items())[:_SUMMARY_TOP_N]:
            class_rows.append(
                f"""
                <div class="flex justify-between text-sm py-1">
                    <span class="text-gray-600">{html.escape(cls)}</span>
                    <span class="font-semibold">{count}</span>
                </div>
                """
            )

        html_content = f"""
        <div class="space-y-4">
            <div>
                <p class="text-sm font-medium text-gray-700 mb-2">By Model</p>
                <div class="space-y-1">
                    {"".join(model_rows) if model_rows else '<p class="text-sm text-gray-500">No data</p>'}
                </div>
            </div>
            <div>
                <p class="text-sm font-medium text-gray-700 mb-2">By Class</p>
                <div class="space-y-1">
                    {"".join(class_rows) if class_rows else '<p class="text-sm text-gray-500">No data</p>'}
                </div>
            </div>
        </div>
        """
        return HTMLResponse(html_content)

    # Return JSON for API requests
    return AnalyticsSummaryResponse(
        total_predictions=summary["total_predictions"],
        predictions_by_model=summary["predictions_by_model"],
        predictions_by_class=summary["predictions_by_class"],
        average_confidence=summary["average_confidence"],
        average_confidence_by_model=summary["average_confidence_by_model"],
    )


@router.get("/analytics/performance", response_model=ModelPerformanceResponse)
async def get_performance(
    start_date: datetime | None = Query(None, description="Filter start date"),
    end_date: datetime | None = Query(None, description="Filter end date"),
    db: Session = Depends(get_db),
) -> ModelPerformanceResponse:
    """Get detailed model performance comparison.

    Provides per-model statistics for performance analysis and comparison.

    Args:
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        db: Database session

    Returns:
        ModelPerformanceResponse: Performance metrics by model
    """
    query = db.query(PredictionHistory)

    if start_date:
        query = query.filter(PredictionHistory.timestamp >= start_date)

    if end_date:
        query = query.filter(PredictionHistory.timestamp <= end_date)

    # Get performance metrics by model in a single query
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

    # Fetch all class distributions in one grouped query (avoids N+1)
    all_class_distributions = (
        query.with_entities(
            PredictionHistory.model_name,
            PredictionHistory.predicted_class,
            func.count(PredictionHistory.id).label("count"),
        )
        .group_by(PredictionHistory.model_name, PredictionHistory.predicted_class)
        .all()
    )
    class_dist_by_model: dict[str, dict[str, int]] = defaultdict(dict)
    for mn, predicted_class, count in all_class_distributions:
        class_dist_by_model[mn][predicted_class] = count

    models_data: dict[str, dict[str, Any]] = {}
    for model_name, total, avg_conf, min_conf, max_conf in model_stats:
        models_data[model_name] = {
            "total_predictions": total,
            "average_confidence": float(avg_conf) if avg_conf else 0.0,
            "min_confidence": float(min_conf) if min_conf else 0.0,
            "max_confidence": float(max_conf) if max_conf else 0.0,
            "class_distribution": class_dist_by_model.get(model_name, {}),
        }

    return ModelPerformanceResponse(models=models_data)
