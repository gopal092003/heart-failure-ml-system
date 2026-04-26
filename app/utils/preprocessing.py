import numpy as np

from app.utils.feature_order import align_features


def prepare_model_input(input_data: dict, feature_order: list):
    """
    Convert API input dict into model-ready numpy array.
    """
    ordered_values = align_features(input_data, feature_order)

    return np.array([ordered_values])