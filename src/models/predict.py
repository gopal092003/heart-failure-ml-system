import numpy as np


def predict(model, X):
    """
    Predict class labels.
    """
    return model.predict(X)


def predict_proba(model, X):
    """
    Predict probabilities.
    """
    return model.predict_proba(X)[:, 1]


def predict_with_threshold(model, X, threshold: float):
    """
    Custom threshold prediction.
    """
    probs = predict_proba(model, X)
    return (probs > threshold).astype(int)


def prepare_input(input_data: dict, feature_order: list):
    """
    Convert API input to model-ready array.
    """
    return np.array([[input_data[f] for f in feature_order]])