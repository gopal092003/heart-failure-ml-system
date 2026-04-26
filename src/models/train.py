import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


def build_model(config: dict):
    """
    Build model from config.
    """
    model_cfg = config["model"]

    model = RandomForestClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        min_samples_split=model_cfg["min_samples_split"],
        class_weight=model_cfg.get("class_weight"),
        n_jobs=model_cfg.get("n_jobs", -1),
        random_state=model_cfg["random_state"]
    )

    return model


def train_model(model, X_train, y_train):
    """
    Train model.
    """
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str):
    """
    Save trained model.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, file_path)


def load_model(path: str):
    """
    Load model from disk.
    """
    return joblib.load(path)