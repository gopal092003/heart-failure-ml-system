from sklearn.model_selection import GridSearchCV


def tune_model(model, X_train, y_train):
    """
    Perform hyperparameter tuning.
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best Params:", grid.best_params_)

    return grid.best_estimator_