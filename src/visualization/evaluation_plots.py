import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from pathlib import Path


def save_plot(fig, path: str):
    """
    Save matplotlib figure.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_test, y_prob, path: str):
    """
    Plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    save_plot(fig, path)


def plot_confusion_matrix(model, X_test, y_test, path: str):
    """
    Plot confusion matrix.
    """
    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)

    ax.set_title("Confusion Matrix")

    save_plot(fig, path)


def plot_calibration_curve(y_test, y_prob, path: str):
    """
    Plot calibration curve.
    """
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

    fig, ax = plt.subplots()

    ax.plot(prob_pred, prob_true, marker='o')
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Probability")
    ax.set_title("Calibration Curve")

    save_plot(fig, path)