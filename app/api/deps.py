import json
import numpy as np
import joblib
import shap

_model = None
_feature_order = None
_explainer = None
_schema = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load("models/trained/model_v1.pkl")
    return _model


def get_feature_order():
    global _feature_order
    if _feature_order is None:
        with open("models/artifacts/feature_names.json") as f:
            _feature_order = json.load(f)["features"]
    return _feature_order


def get_schema():
    global _schema
    if _schema is None:
        with open("data/validation/schema.json") as f:
            _schema = json.load(f)
    return _schema


def get_explainer():
    global _explainer
    if _explainer is None:
        model = get_model()
        background = np.load("models/artifacts/shap_background.npy")
        _explainer = shap.TreeExplainer(model, data=background)
    return _explainer