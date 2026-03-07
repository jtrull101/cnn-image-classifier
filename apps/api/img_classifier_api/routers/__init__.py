"""API routers package."""

from img_classifier_api.routers import analytics, datasets, history, models, predictions

__all__ = ["models", "predictions", "history", "analytics", "datasets"]
