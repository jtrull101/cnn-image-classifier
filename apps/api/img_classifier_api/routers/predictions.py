"""Router for prediction endpoints."""

import asyncio
import base64
import hashlib
import logging
import os
import uuid
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from img_classifier_api.crud.predictions import create_prediction
from img_classifier_api.database import get_db
from img_classifier_api.schemas import PredictionResponse


logger = logging.getLogger(__name__)

# Maximum accepted upload size in bytes.  Override with IMG_CLASSIFIER_MAX_UPLOAD_BYTES.
_DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_THUMBNAIL_SIZE = (100, 100)
_MAX_BATCH_FILES = 100
_DEFAULT_FILENAME = "unknown.jpg"

try:
    _MAX_UPLOAD_BYTES: int = int(
        os.environ.get("IMG_CLASSIFIER_MAX_UPLOAD_BYTES", _DEFAULT_MAX_UPLOAD_BYTES)
    )
except ValueError:
    _MAX_UPLOAD_BYTES = _DEFAULT_MAX_UPLOAD_BYTES


class ComparePredictionResponse(BaseModel):
    """Response for multi-model comparison."""

    image_name: str
    predictions: dict[str, PredictionResponse]


router = APIRouter()


# ---------------------------------------------------------------------------
# Core prediction helper (no HTTP, no DB — reusable by batch/compare)
# ---------------------------------------------------------------------------


async def _run_prediction(
    request: Request,
    file: UploadFile,
    model_name: str | None,
) -> tuple[PredictionResponse, bytes, np.ndarray]:
    """Run inference on *file* and return the response plus raw bytes and image array.

    Returns:
        (PredictionResponse, raw_contents_bytes, decoded_image_ndarray)

    Raises:
        HTTPException on validation or inference errors.
    """
    model_manager = request.app.state.model_manager

    # Validate MIME type (client-supplied, so only a first-pass check)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Read with size guard (before model lookup so 413 is always reachable)
    max_size = _MAX_UPLOAD_BYTES
    contents = await file.read(max_size + 1)
    if len(contents) > max_size:
        raise HTTPException(
            413,
            f"File too large. Maximum allowed size is {max_size // (1024 * 1024)} MB.",
        )

    # Get model and info
    try:
        model = model_manager.get_model(model_name)
        info = model_manager.get_info(model_name)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e

    # Decode image (validates actual file headers, not just MIME)
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Unable to decode image. Ensure the file is a valid image.")

    # Resize to model input shape
    input_shape = tuple(info.input_shape[:2])
    image_resized = cv2.resize(image, input_shape)

    # Normalize and predict
    image_array = np.asarray(image_resized, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    raw_predictions = await asyncio.to_thread(model.predict, image_batch, verbose=0)
    probabilities = raw_predictions[0]

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = info.class_names[predicted_idx]
    confidence = float(probabilities[predicted_idx])
    prob_dict = {
        class_name: float(prob)
        for class_name, prob in zip(info.class_names, probabilities, strict=False)
    }

    response = PredictionResponse(
        class_name=predicted_class,
        confidence=confidence,
        probabilities=prob_dict,
        model_name=info.name,
    )
    return response, contents, image


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    model_name: str | None = None,
    save_to_history: bool = True,
    user_session: str | None = None,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Predict the class of an uploaded image.

    Args:
        request: FastAPI request object
        file: Image file to classify
        model_name: Optional model name (uses current model if not specified)
        save_to_history: Whether to save prediction to database (default: True)
        user_session: Optional user session identifier for tracking
        db: Database session (injected by FastAPI)

    Returns:
        PredictionResponse with class prediction and confidence
    """
    try:
        result, contents, image = await _run_prediction(request, file, model_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e!s}") from e

    if save_to_history:
        info = request.app.state.model_manager.get_info(model_name)
        image_hash = hashlib.sha256(contents).hexdigest()
        thumbnail = cv2.resize(image, _THUMBNAIL_SIZE)
        _, buffer = cv2.imencode(".jpg", thumbnail)
        thumbnail_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        create_prediction(
            db,
            {
                "image_name": file.filename or _DEFAULT_FILENAME,
                "image_hash": image_hash,
                "model_name": info.name,
                "predicted_class": result.class_name,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "image_thumbnail": thumbnail_b64,
                "user_session": user_session,
            },
        )

    return result


@router.post("/predict/batch")
async def batch_predict(
    request: Request,
    files: list[UploadFile] = File(...),
    model_name: str | None = None,
    save_to_history: bool = True,
    user_session: str | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Batch prediction endpoint with WebSocket progress updates.

    Args:
        request: FastAPI request object
        files: List of image files to classify
        model_name: Optional model name
        save_to_history: Whether to save predictions to database
        user_session: Optional user session identifier
        db: Database session (injected by FastAPI)

    Returns:
        dict: Summary of batch predictions
    """
    ws_manager = request.app.state.ws_manager
    total_files = len(files)
    batch_id = str(uuid.uuid4())
    results = []
    errors = []

    if total_files > _MAX_BATCH_FILES:
        raise HTTPException(400, f"Too many files. Maximum batch size is {_MAX_BATCH_FILES}.")

    if ws_manager:
        await ws_manager.send_batch_progress(batch_id, 0, total_files, "Starting batch processing")

    for idx, file in enumerate(files):
        try:
            result, contents, image = await _run_prediction(request, file, model_name)

            if save_to_history:
                info = request.app.state.model_manager.get_info(model_name)
                image_hash = hashlib.sha256(contents).hexdigest()
                thumbnail = cv2.resize(image, _THUMBNAIL_SIZE)
                _, buffer = cv2.imencode(".jpg", thumbnail)
                thumbnail_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
                create_prediction(
                    db,
                    {
                        "image_name": file.filename or _DEFAULT_FILENAME,
                        "image_hash": image_hash,
                        "model_name": info.name,
                        "predicted_class": result.class_name,
                        "confidence": result.confidence,
                        "probabilities": result.probabilities,
                        "image_thumbnail": thumbnail_b64,
                        "user_session": user_session,
                    },
                )

            results.append({"filename": file.filename, "prediction": result.model_dump()})

            if ws_manager:
                await ws_manager.send_batch_progress(
                    batch_id,
                    idx + 1,
                    total_files,
                    f"Processed {file.filename}",
                )
        except Exception as e:
            logger.warning("Batch item %r failed: %s", file.filename, e)
            errors.append({"filename": file.filename, "error": str(e)})

    if ws_manager:
        await ws_manager.send_batch_progress(
            batch_id, total_files, total_files, "Batch processing complete"
        )

    return {
        "total": total_files,
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


@router.post("/predict/compare", response_model=ComparePredictionResponse)
async def compare_predictions(
    request: Request,
    file: UploadFile = File(...),
    model_names: list[str] | None = Query(None),
) -> ComparePredictionResponse:
    """Compare predictions across multiple models for a single image.

    Args:
        request: FastAPI request object
        file: Image file to classify
        model_names: List of model names to compare (uses all loaded if not specified)

    Returns:
        ComparePredictionResponse: Predictions from each model
    """
    model_manager = request.app.state.model_manager

    if model_names is None:
        model_names = list(model_manager.models.keys())

    if not model_names:
        raise HTTPException(400, "No models available for comparison")

    predictions: dict[str, PredictionResponse] = {}

    for model_name in model_names:
        try:
            await file.seek(0)
            result, _, _ = await _run_prediction(request, file, model_name)
            predictions[model_name] = result
        except Exception:
            logger.exception("Inference failed for model '%s' during comparison", model_name)
            predictions[model_name] = PredictionResponse(
                class_name="ERROR",
                confidence=0.0,
                probabilities={},
                model_name=model_name,
            )

    return ComparePredictionResponse(
        image_name=file.filename or _DEFAULT_FILENAME,
        predictions=predictions,
    )
