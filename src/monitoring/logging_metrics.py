import json
from pathlib import Path
from datetime import datetime


def log_prediction(input_data: dict, prediction: float, path: str):
    """
    Log individual prediction (for monitoring).
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction
    }

    with open(file_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_batch_predictions(predictions: list, path: str):
    """
    Log batch predictions.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "predictions": predictions
    }

    with open(file_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_error(error_message: str, path: str):
    """
    Log errors during inference.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "error": error_message
    }

    with open(file_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_drift(drift_results: dict, path: str):
    """
    Log drift detection results.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "drift": drift_results
    }

    with open(file_path, "a") as f:
        f.write(json.dumps(record) + "\n")