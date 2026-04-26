import numpy as np
import pandas as pd

from src.models.train import load_model
from src.explainability.shap_explainer import create_explainer, compute_shap_values
from src.explainability.shap_export import save_summary_plot
from src.utils.config_loader import load_config


def run_explain():
    config = load_config()

    # Load model
    model = load_model("models/trained/model_v1.pkl")

    # Load test data
    df = pd.read_parquet("data/processed/test.parquet")

    X_test = df.drop(config["data"]["target_column"], axis=1)

    # Load background
    background = np.load("models/artifacts/shap_background.npy")

    # SHAP
    explainer = create_explainer(model, background)
    shap_values = compute_shap_values(explainer, X_test)

    save_summary_plot(
        shap_values,
        X_test,
        "reports/shap/summary_from_script.png"
    )

    print("✅ SHAP explanation generated!")


if __name__ == "__main__":
    run_explain()