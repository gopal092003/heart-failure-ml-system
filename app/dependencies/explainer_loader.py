import numpy as np
import shap
from functools import lru_cache
from app.core.settings import settings
from app.dependencies.model_loader import load_model


@lru_cache()
def load_explainer():
    """
    Load SHAP explainer once.
    """
    model = load_model()
    background = np.load(settings.SHAP_BACKGROUND_PATH)

    explainer = shap.TreeExplainer(model, data=background)
    return explainer