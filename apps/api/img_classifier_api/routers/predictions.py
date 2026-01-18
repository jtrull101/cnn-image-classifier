"""Router for prediction endpoints."""

import base64
import hashlib
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from img_classifier_api.crud.predictions import create_prediction
from img_classifier_api.database import get_db


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    model_name: str = Field(..., description="Name of the model used")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    image_count: int = Field(..., description="Number of images to process")
    model_name: Optional[str] = Field(None, description="Model to use (defaults to current)")


class ComparePredictionResponse(BaseModel):
    """Response for multi-model comparison."""

    image_name: str
    predictions: Dict[str, PredictionResponse]


router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    save_to_history: bool = True,
    user_session: Optional[str] = None,
):
    """
    Predict the class of an uploaded image.

    Args:
        request: FastAPI request object
        file: Image file to classify
        model_name: Optional model name (uses current model if not specified)
        save_to_history: Whether to save prediction to database (default: True)
        user_session: Optional user session identifier for tracking

    Returns:
        PredictionResponse with class prediction and confidence
    """
    model_manager = request.app.state.model_manager
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Get model and info
    try:
        model = model_manager.get_model(model_name)
        info = model_manager.get_info(model_name)
    except ValueError as e:
        raise HTTPException(404, str(e))

    # Read and process image
    try:
        # Read file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(400, "Unable to decode image")

        # Resize to model input shape
        input_shape = tuple(info.input_shape[:2])
        image_resized = cv2.resize(image, input_shape)

        # Normalize
        image_array = np.asarray(image_resized, dtype=np.float32) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

        # Predict
        predictions = model.predict(image_batch, verbose=0)
        probabilities = predictions[0]

        # Get top prediction
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = info.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # Build probability dict
        prob_dict = {
            class_name: float(prob) for class_name, prob in zip(info.class_names, probabilities)
        }

        # Save to database if requested
        if save_to_history:
            # Generate image hash
            image_hash = hashlib.sha256(contents).hexdigest()

            # Create thumbnail (base64 encoded)
            thumbnail_size = (100, 100)
            thumbnail = cv2.resize(image, thumbnail_size)
            _, buffer = cv2.imencode(".jpg", thumbnail)
            thumbnail_b64 = base64.b64encode(buffer).decode("utf-8")

            # Save to database
            db: Session = next(get_db())
            try:
                create_prediction(
                    db,
                    {
                        "image_name": file.filename or "unknown.jpg",
                        "image_hash": image_hash,
                        "model_name": info.name,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "probabilities": prob_dict,
                        "image_thumbnail": thumbnail_b64,
                        "user_session": user_session,
                    },
                )
            finally:
                db.close()

        return PredictionResponse(
            class_name=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            model_name=info.name,
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")


@router.post("/predict/batch")
async def batch_predict(
    request: Request,
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = None,
    save_to_history: bool = True,
    user_session: Optional[str] = None,
):
    """
    Batch prediction endpoint with WebSocket progress updates.

    Args:
        request: FastAPI request object
        files: List of image files to classify
        model_name: Optional model name
        save_to_history: Whether to save predictions to database
        user_session: Optional user session identifier

    Returns:
        dict: Summary of batch predictions
    """
    ws_manager = request.app.state.ws_manager
    total_files = len(files)
    results = []
    errors = []

    # Send initial batch progress
    if ws_manager:
        await ws_manager.send_batch_progress(0, total_files, "Starting batch processing")

    for idx, file in enumerate(files):
        try:
            # Process each image
            result = await predict_image(
                request=request,
                file=file,
                model_name=model_name,
                save_to_history=save_to_history,
                user_session=user_session,
            )
            results.append({"filename": file.filename, "prediction": result.model_dump()})

            # Send progress update
            if ws_manager:
                progress = ((idx + 1) / total_files) * 100
                await ws_manager.send_batch_progress(
                    idx + 1,
                    total_files,
                    f"Processed {file.filename}",
                    progress=progress,
                )

        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    # Send completion message
    if ws_manager:
        await ws_manager.send_batch_progress(
            total_files, total_files, "Batch processing complete", progress=100
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
    model_names: Optional[List[str]] = None,
):
    """
    Compare predictions across multiple models for a single image.

    Args:
        request: FastAPI request object
        file: Image file to classify
        model_names: List of model names to compare (uses all loaded if not specified)

    Returns:
        ComparePredictionResponse: Predictions from each model
    """
    model_manager = request.app.state.model_manager

    if model_names is None:
        # Use all loaded models
        model_names = list(model_manager.models.keys())

    if not model_names:
        raise HTTPException(400, "No models available for comparison")

    predictions: Dict[str, PredictionResponse] = {}

    for model_name in model_names:
        try:
            # Reset file pointer for each prediction
            await file.seek(0)

            # Get prediction
            result = await predict_image(
                request=request,
                file=file,
                model_name=model_name,
                save_to_history=False,  # Don't save comparison predictions
            )
            predictions[model_name] = result

        except Exception:
            # Include error in response - create PredictionResponse with error values
            predictions[model_name] = PredictionResponse(
                class_name="ERROR",
                confidence=0.0,
                probabilities={},
                model_name=model_name,
            )

    return ComparePredictionResponse(
        image_name=file.filename or "unknown.jpg",
        predictions=predictions,
    )
