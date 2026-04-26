import json
from app.core.settings import settings


def load_feature_order():
    """
    Load feature order from saved artifact.
    """
    with open(settings.FEATURE_PATH, "r") as f:
        data = json.load(f)

    return data["features"]


def align_features(input_data: dict, feature_order: list):
    """
    Reorder input dictionary into correct feature order.
    """
    return [input_data[feature] for feature in feature_order]