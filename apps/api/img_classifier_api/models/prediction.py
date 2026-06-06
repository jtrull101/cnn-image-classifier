"""Prediction history database model.

Stores prediction results for tracking and analytics.
"""

import json
from typing import Any, cast

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.sql import func

from img_classifier_api.database import Base


class PredictionHistory(Base):
    """Model for storing prediction history.

    Attributes:
        id: Primary key
        timestamp: When the prediction was made
        image_name: Name of the uploaded image
        image_hash: Hash of the image for deduplication
        model_name: Name of the model used
        predicted_class: The predicted class name
        confidence: Confidence score (0-1)
        probabilities: JSON string of all class probabilities
        image_thumbnail: Optional base64 encoded thumbnail
        user_session: Optional session identifier for multi-user support
    """

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    image_name = Column(String(255), nullable=False)
    image_hash = Column(String(64), index=True)
    model_name = Column(String(255), nullable=False, index=True)
    predicted_class = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(Text, nullable=False)  # JSON string
    image_thumbnail = Column(Text, nullable=True)  # Base64 encoded image
    user_session = Column(String(255), nullable=True, index=True)

    # Indexes for common queries
    __table_args__ = (
        Index("idx_timestamp_model", "timestamp", "model_name"),
        Index("idx_predicted_class", "predicted_class"),
    )

    def get_probabilities(self) -> dict[str, float]:
        """Parse probabilities JSON string to dictionary."""
        try:
            # Cast to str for type checker - at runtime this is always a string
            probs_str = cast(str, self.probabilities) if self.probabilities is not None else "{}"
            return json.loads(probs_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary for API responses."""
        # Cast timestamp to datetime for type checker - at runtime this is always datetime
        timestamp_val = cast(Any, self.timestamp)
        return {
            "id": self.id,
            "timestamp": timestamp_val.isoformat() if timestamp_val is not None else None,
            "image_name": self.image_name,
            "image_hash": self.image_hash,
            "model_name": self.model_name,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.get_probabilities(),
            "image_thumbnail": self.image_thumbnail,
            "user_session": self.user_session,
        }

    def __repr__(self) -> str:
        """Return a debug representation of the prediction row."""
        return (
            f"<PredictionHistory(id={self.id}, "
            f"image={self.image_name}, "
            f"model={self.model_name}, "
            f"class={self.predicted_class}, "
            f"confidence={self.confidence:.2f})>"
        )
