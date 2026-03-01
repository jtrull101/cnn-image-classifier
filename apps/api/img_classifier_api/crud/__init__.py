"""CRUD operations package."""

from img_classifier_api.crud.predictions import (
    create_prediction,
    delete_prediction,
    export_predictions,
    get_analytics_summary,
    get_prediction_by_id,
    get_predictions,
    get_predictions_by_model,
)

__all__ = [
    "create_prediction",
    "get_predictions",
    "get_prediction_by_id",
    "delete_prediction",
    "get_predictions_by_model",
    "export_predictions",
    "get_analytics_summary",
]
