"""Canonical JSON stubs used by page.route() to intercept API calls in e2e tests."""

MODELS_EMPTY: dict = {
    "models": [],
    "current_model": None,
}

MODELS_ONE_LOADED: dict = {
    "models": [
        {
            "name": "alzheimer_cnn_v1",
            "path": "/models/alzheimer_cnn_v1.keras",
            "num_classes": 4,
            "class_names": [
                "MildDemented",
                "ModerateDemented",
                "NonDemented",
                "VeryMildDemented",
            ],
            "input_shape": [176, 208, 3],
            "accuracy": 0.947,
        }
    ],
    "current_model": "alzheimer_cnn_v1",
}

MODELS_TWO_LOADED: dict = {
    "models": [
        {
            "name": "alzheimer_cnn_v1",
            "path": "/models/alzheimer_cnn_v1.keras",
            "num_classes": 4,
            "class_names": [
                "MildDemented",
                "ModerateDemented",
                "NonDemented",
                "VeryMildDemented",
            ],
            "input_shape": [176, 208, 3],
            "accuracy": 0.947,
        },
        {
            "name": "alzheimer_cnn_v2",
            "path": "/models/alzheimer_cnn_v2.keras",
            "num_classes": 4,
            "class_names": [
                "MildDemented",
                "ModerateDemented",
                "NonDemented",
                "VeryMildDemented",
            ],
            "input_shape": [176, 208, 3],
            "accuracy": 0.965,
        },
    ],
    "current_model": "alzheimer_cnn_v1",
}

ANALYTICS_SUMMARY_EMPTY: dict = {
    "total_predictions": 0,
    "predictions_by_model": {},
    "predictions_by_class": {},
    "average_confidence": 0.0,
    "average_confidence_by_model": {},
}

ANALYTICS_SUMMARY_POPULATED: dict = {
    "total_predictions": 42,
    "predictions_by_model": {"alzheimer_cnn_v1": 42},
    "predictions_by_class": {
        "NonDemented": 20,
        "MildDemented": 12,
        "ModerateDemented": 7,
        "VeryMildDemented": 3,
    },
    "average_confidence": 0.891,
    "average_confidence_by_model": {"alzheimer_cnn_v1": 0.891},
}

HISTORY_EMPTY: dict = {
    "predictions": [],
    "total": 0,
    "page": 1,
    "per_page": 20,
}

PREDICTION_RESULT: dict = {
    "class_name": "NonDemented",
    "confidence": 0.921,
    "probabilities": {
        "MildDemented": 0.042,
        "ModerateDemented": 0.015,
        "NonDemented": 0.921,
        "VeryMildDemented": 0.022,
    },
    "model_name": "alzheimer_cnn_v1",
}

DISCOVER_MODELS_RESPONSE: dict = {
    "model_files": [
        "/models/alzheimer_cnn_v1.keras",
        "/models/alzheimer_cnn_v2.keras",
    ]
}
