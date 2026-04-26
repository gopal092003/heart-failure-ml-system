import json
import numpy as np
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger, get_logger
from src.utils.reproducibility import set_seed

from src.data.load import load_raw_data
from src.data.validate import validate_raw_data
from src.data.preprocess import preprocess_pipeline

from src.features.build import (
    build_preprocessor,
    apply_preprocessing,
    get_feature_names
)
from src.features.feature_checks import run_feature_checks

from src.models.train import build_model, train_model, save_model
from src.models.evaluate import evaluate_model, save_metrics
from src.models.calibrate import calibrate_model

from src.explainability.shap_explainer import (
    create_explainer,
    compute_shap_values
)
from src.explainability.shap_export import save_summary_plot


def ensure_dirs():
    """Ensure required directories exist."""
    dirs = [
        "models/trained",
        "models/calibrated",
        "models/artifacts",
        "models/metadata",
        "reports/shap"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_feature_names(feature_names):
    """Save feature order for inference."""
    path = "models/artifacts/feature_names.json"
    with open(path, "w") as f:
        json.dump({"features": feature_names}, f, indent=4)


def save_shap_background(background):
    """Save SHAP background dataset."""
    path = "models/artifacts/shap_background.npy"
    np.save(path, background)


def save_model_params(config):
    """Save model parameters."""
    path = "models/metadata/params.json"
    with open(path, "w") as f:
        json.dump(config["model"], f, indent=4)


def run_training_pipeline():
    # =========================
    # Setup
    # =========================
    config = load_config()
    setup_logger()
    logger = get_logger()

    set_seed(config["model"]["random_state"])
    ensure_dirs()

    logger.info("🚀 Starting training pipeline...")

    # =========================
    # Load + Validate
    # =========================
    df = load_raw_data(config["data"]["raw_path"])
    validate_raw_data(df)

    # =========================
    # Preprocessing
    # =========================
    X_train, X_test, y_train, y_test = preprocess_pipeline(df, config)

    # =========================
    # Feature Engineering
    # =========================
    preprocessor = build_preprocessor(config)

    X_train_t, X_test_t = apply_preprocessing(
        preprocessor, X_train, X_test
    )

    run_feature_checks(X_train, X_test, df, config)

    feature_names = get_feature_names(preprocessor)

    # =========================
    # Model Training
    # =========================
    model = build_model(config)
    model = train_model(model, X_train_t, y_train)

    # =========================
    # Calibration (optional)
    # =========================
    calibrated_model = calibrate_model(
        model, X_train_t, y_train, config
    )

    # =========================
    # Evaluation
    # =========================
    metrics = evaluate_model(calibrated_model, X_test_t, y_test)

    save_metrics(metrics, "models/metadata/metrics.json")
    save_model_params(config)

    # =========================
    # Save Models
    # =========================
    save_model(model, "models/trained/model_v1.pkl")

    if config["calibration"]["enabled"]:
        save_model(
            calibrated_model,
            "models/calibrated/calibrated_model_v1.pkl"
        )
        final_model = calibrated_model
    else:
        final_model = model

    logger.info("💾 Models saved successfully")

    # =========================
    # Save Artifacts
    # =========================
    save_feature_names(feature_names)

    background_size = config["shap"]["background_sample_size"]
    background = X_train_t[:background_size]

    save_shap_background(background)

    logger.info("📦 Artifacts saved (feature names + SHAP background)")

    # =========================
    # SHAP Analysis
    # =========================
    logger.info("🔍 Running SHAP analysis...")

    explainer = create_explainer(final_model, background)
    shap_values = compute_shap_values(explainer, X_test_t)

    save_summary_plot(
        shap_values,
        X_test_t,
        "reports/shap/summary.png"
    )

    logger.info("📊 SHAP summary plot saved")

    logger.info("✅ Training pipeline completed successfully!")


if __name__ == "__main__":
    run_training_pipeline()