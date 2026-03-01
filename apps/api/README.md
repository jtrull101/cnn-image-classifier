# FastAPI Image Classification API

Modern FastAPI-based REST API for generic image classification with support for multiple models.

## Features

- **Multi-Model Support**: Load and switch between different models
- **Generic Classification**: Works with any trained image classification model
- **Modern UI**: Responsive web interface with drag-and-drop upload
- **REST API**: Full-featured API with automatic documentation
- **Auto-Discovery**: Automatically finds and loads best available model
- **Model Metadata**: Extracts class names, accuracy, and input shape
- **Real-time Predictions**: Fast inference with confidence scores

## Quick Start

### Install Dependencies

```bash
cd apps/api
uv pip install -e .
```

### Run the Server

```bash
# Development mode (with auto-reload)
python -m img_classifier_api.app

# Or with uvicorn directly
uvicorn img_classifier_api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Web Interface**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### Web Interface

- **GET /** - Main web interface with drag-and-drop image upload

### Prediction

- **POST /api/predict** - Classify an image
  - Upload file as multipart/form-data
  - Optional: `model_name` query parameter to use specific model
  - Returns: Class prediction, confidence, and probabilities

Example:
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/predict",
        files={"file": f}
    )
    result = response.json()
    print(f"Predicted: {result['class_name']} ({result['confidence']:.2%})")
```

### Model Management

- **GET /api/models** - List all loaded models
- **GET /api/models/{model_name}** - Get info about specific model
- **POST /api/models/{model_name}/activate** - Set model as current
- **POST /api/models/load** - Load a new model from file path

### Health Check

- **GET /health** - Service health status

## Model Discovery

The API automatically searches for models in:
1. `/tmp/img_classifier/models/`
2. `/tmp/img_classifier_cnn/models/`
3. `./models/` (current directory)

Supported formats: `.keras`, `.h5`

## Model Metadata

To provide class names and accuracy, create a JSON file with the same name as your model:

**Example: `my_model.keras` + `my_model.json`**

```json
{
  "class_names": ["cat", "dog", "bird"],
  "accuracy": 0.95,
  "description": "Pet classifier trained on 10k images"
}
```

The API will automatically:
- Extract accuracy from filename (e.g., `model_95%.keras`)
- Load metadata from accompanying JSON file
- Use generic class names if no metadata available

## Usage Examples

### Web Interface

1. Open http://localhost:8000 in your browser
2. Drag and drop an image or click to upload
3. Click "Classify Image"
4. View results with confidence scores and probabilities

### Python Client

```python
import requests
from pathlib import Path

# Predict an image
def predict_image(image_path: Path):
    with open(image_path, "rb") as f:
        response = requests.post(
            "http://localhost:8000/api/predict",
            files={"file": ("image.jpg", f, "image/jpeg")}
        )
    return response.json()

# List available models
def list_models():
    response = requests.get("http://localhost:8000/api/models")
    return response.json()

# Switch model
def set_model(model_name: str):
    response = requests.post(
        f"http://localhost:8000/api/models/{model_name}/activate"
    )
    return response.json()

# Example usage
result = predict_image(Path("cat.jpg"))
print(f"Class: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### cURL Examples

```bash
# Predict
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@image.jpg"

# List models
curl "http://localhost:8000/api/models"

# Get model info
curl "http://localhost:8000/api/models/my_model"

# Activate model
curl -X POST "http://localhost:8000/api/models/my_model/activate"

# Health check
curl "http://localhost:8000/health"
```

## Response Formats

### Prediction Response

```json
{
  "class_name": "cat",
  "confidence": 0.95,
  "probabilities": {
    "cat": 0.95,
    "dog": 0.03,
    "bird": 0.02
  },
  "model_name": "pet_classifier"
}
```

### Model Info Response

```json
{
  "name": "pet_classifier",
  "path": "/tmp/models/pet_classifier_95%.keras",
  "num_classes": 3,
  "class_names": ["cat", "dog", "bird"],
  "input_shape": [128, 128, 3],
  "accuracy": 0.95
}
```

## Configuration

The API can be configured through environment variables or by modifying `app.py`:

```python
# Model search directories
model_manager.default_model_dirs = [
    Path("/custom/model/dir"),
    Path("/another/dir"),
]
```

## Deployment

### Production with Gunicorn

```bash
gunicorn img_classifier_api.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

CMD ["uvicorn", "img_classifier_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Systemd Service

```ini
[Unit]
Description=Image Classification API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/app
ExecStart=/path/to/venv/bin/uvicorn img_classifier_api.app:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Features Comparison: Flask vs FastAPI

| Feature | Old (Flask) | New (FastAPI) |
|---------|-------------|---------------|
| Framework | Flask | FastAPI |
| API Docs | Manual | Automatic (Swagger/ReDoc) |
| Type Validation | Manual | Automatic (Pydantic) |
| Async Support | No | Yes |
| Multi-Model | No | Yes |
| Model Discovery | Manual | Automatic |
| Class Names | Hardcoded | Dynamic |
| Frontend | Basic HTML | Modern Responsive UI |
| Error Handling | Basic | Comprehensive |
| Performance | Good | Excellent |

## Troubleshooting

### No Models Found

```bash
# Check model directories exist
ls /tmp/img_classifier/models/
ls /tmp/img_classifier_cnn/models/

# Or place models in current directory
mkdir -p models
cp your_model.keras models/
```

### Model Loading Errors

- Ensure model is saved in `.keras` or `.h5` format
- Check TensorFlow/Keras version compatibility
- Verify model file is not corrupted

### Port Already in Use

```bash
# Use different port
uvicorn img_classifier_api.app:app --port 8001

# Or find and kill process using port 8000
lsof -ti:8000 | xargs kill -9  # Unix/Mac
netstat -ano | findstr :8000   # Windows
```

## Development

### Run Tests

```bash
pytest apps/api/tests/
```

### Format Code

```bash
ruff format apps/api/
```

### Type Check

```bash
pyright apps/api/
```

## Contributing

When adding new features:
1. Update API documentation
2. Add tests for new endpoints
3. Update this README
4. Follow existing code style

## License

MIT License - See LICENSE file for details

