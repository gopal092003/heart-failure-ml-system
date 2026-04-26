import pandas as pd

from src.models.train import load_model, save_model
from src.models.calibrate import calibrate_model
from src.utils.config_loader import load_config


def run_calibration():
    config = load_config()

    model = load_model("models/trained/model_v1.pkl")

    df = pd.read_parquet("data/processed/train.parquet")

    X = df.drop(config["data"]["target_column"], axis=1)
    y = df[config["data"]["target_column"]]

    calibrated_model = calibrate_model(model, X, y, config)

    save_model(
        calibrated_model,
        "models/calibrated/calibrated_model_v1.pkl"
    )

    print("✅ Model calibrated and saved!")


if __name__ == "__main__":
    run_calibration()