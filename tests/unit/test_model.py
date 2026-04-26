import numpy as np
from src.models.train import build_model
from src.models.predict import predict_proba


def test_model_training():
    config = {
        "model": {
            "n_estimators": 10,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42
        }
    }

    model = build_model(config)

    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)

    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)


def test_predict_proba_range():
    config = {
        "model": {
            "n_estimators": 10,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42
        }
    }

    model = build_model(config)

    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)

    model.fit(X, y)

    probs = predict_proba(model, X)

    assert ((probs >= 0) & (probs <= 1)).all()