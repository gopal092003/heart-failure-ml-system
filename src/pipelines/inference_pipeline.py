import numpy as np

from src.utils.config_loader import load_config
from src.models.predict import predict_proba, prepare_input
from src.validation.input_validation import full_input_validation
from src.explainability.shap_explainer import create_explainer


def run_inference(input_data: dict, model, feature_order: list, schema: dict, explainer=None):
    """
    Full inference pipeline (API-ready).
    """

    # --- Validate input ---
    full_input_validation(input_data, feature_order, schema)

    # --- Prepare input ---
    X = prepare_input(input_data, feature_order)

    # --- Predict ---
    prob = predict_proba(model, X)[0]

    # --- SHAP (optional) ---
    shap_result = None
    if explainer is not None:
        shap_values = explainer.shap_values(X)[1][0]
        shap_result = dict(zip(feature_order, shap_values.tolist()))

    return {
        "prediction": float(prob),
        "explanations": shap_result
    }