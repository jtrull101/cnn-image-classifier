"""Simple startup script for the FastAPI server.

Run with: python run_api.py.
"""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "img_classifier_api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
