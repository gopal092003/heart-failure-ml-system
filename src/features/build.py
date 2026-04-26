import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def get_feature_lists(config: dict):
    """
    Extract feature groups from config.
    """
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    return numeric_features, categorical_features


def build_preprocessor(config: dict):
    """
    Build sklearn ColumnTransformer.

    Returns:
        preprocessor (ColumnTransformer)
    """
    numeric_features, categorical_features = get_feature_lists(config)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", "passthrough", categorical_features)
        ]
    )

    return preprocessor


def apply_preprocessing(preprocessor, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit on train, transform both train and test.

    Returns:
        X_train_transformed, X_test_transformed
    """
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    return X_train_t, X_test_t


def get_feature_names(preprocessor):
    """
    Extract feature names after transformation.
    """
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            feature_names.extend(columns)

    return feature_names