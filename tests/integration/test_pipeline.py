import os
from src.pipelines.training_pipeline import run_training_pipeline


def test_training_pipeline_runs():
    """
    This is a smoke test — ensures pipeline runs without crashing.
    """
    run_training_pipeline()

    # Check model exists
    assert os.path.exists("models/trained/model_v1.pkl")

    # Check artifacts
    assert os.path.exists("models/artifacts/feature_names.json")
    assert os.path.exists("models/artifacts/shap_background.npy")