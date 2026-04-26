import shap
import numpy as np


def create_explainer(model, background_data=None):
    """
    Create SHAP TreeExplainer.

    Args:
        model: trained model
        background_data: optional background dataset

    Returns:
        explainer
    """
    if background_data is not None:
        return shap.TreeExplainer(model, data=background_data)
    return shap.TreeExplainer(model)


def compute_shap_values(explainer, X):
    """
    Compute SHAP values.

    Returns:
        shap_values (for class 1)
    """
    shap_values = explainer.shap_values(X)

    # Binary classification → take class 1
    if isinstance(shap_values, list):
        return shap_values[1]

    return shap_values


def get_feature_importance(shap_values, feature_names):
    """
    Mean absolute SHAP importance.
    """
    importance = np.abs(shap_values).mean(axis=0)

    return dict(zip(feature_names, importance.tolist()))