from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(model, X_train, y_train, config: dict):
    """
    Calibrate model probabilities.
    """
    calib_cfg = config["calibration"]

    if not calib_cfg.get("enabled", False):
        return model

    calibrated_model = CalibratedClassifierCV(
        model,
        method=calib_cfg["method"],
        cv=calib_cfg["cv"]
    )

    calibrated_model.fit(X_train, y_train)

    return calibrated_model