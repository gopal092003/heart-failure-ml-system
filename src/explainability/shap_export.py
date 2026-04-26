import shap
import matplotlib.pyplot as plt
from pathlib import Path


def save_summary_plot(shap_values, X, path: str):
    """
    Save SHAP summary plot.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_beeswarm_plot(shap_values, X, path: str):
    """
    Save beeswarm plot.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_dependence_plot(feature, shap_values, X, path: str, interaction=None):
    """
    Save dependence plot.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    shap.dependence_plot(feature, shap_values, X, interaction_index=interaction, show=False)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_force_plot(explainer, shap_values, X_row, path: str):
    """
    Save SHAP force plot as HTML.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    force_plot = shap.force_plot(
        explainer.expected_value[1],
        shap_values,
        X_row
    )

    shap.save_html(path, force_plot)