import joblib
from functools import lru_cache
from app.core.settings import settings


@lru_cache()
def load_model():
    """
    Load model once and cache it.
    """
    model = joblib.load(settings.MODEL_PATH)
    return model