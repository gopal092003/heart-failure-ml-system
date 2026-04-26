import json
from pathlib import Path
from datetime import datetime


def log_performance(metrics: dict, path: str):
    """
    Append performance metrics with timestamp.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics
    }

    if file_path.exists():
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def load_performance_history(path: str):
    """
    Load historical metrics.
    """
    file_path = Path(path)

    if not file_path.exists():
        return []

    with open(file_path, "r") as f:
        return json.load(f)


def check_performance_drop(history: list, metric: str = "roc_auc", threshold: float = 0.05):
    """
    Detect performance degradation.
    """
    if len(history) < 2:
        return False

    latest = history[-1]["metrics"].get(metric)
    previous = history[-2]["metrics"].get(metric)

    if latest is None or previous is None:
        return False

    drop = previous - latest

    return drop > threshold


def summarize_performance(history: list):
    """
    Summarize performance history.
    """
    summary = {
        "num_runs": len(history),
        "latest": history[-1] if history else None
    }

    return summary