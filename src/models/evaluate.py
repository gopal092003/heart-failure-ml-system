import json
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    """
    Compute evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    return metrics


def save_metrics(metrics: dict, path: str):
    """
    Save metrics to JSON.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)