import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning (extendable).

    - removes duplicates
    - resets index
    """
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_features_target(df: pd.DataFrame, config: dict):
    """
    Split into X and y.

    Args:
        df (pd.DataFrame)
        config (dict)

    Returns:
        X, y
    """
    target = config["data"]["target_column"]
    drop_cols = config["data"].get("drop_columns", [])

    X = df.drop([target] + drop_cols, axis=1)
    y = df[target]

    return X, y


def split_train_test(X, y, config: dict):
    """
    Train-test split using config.

    Returns:
        X_train, X_test, y_train, y_test
    """
    split_cfg = config["split"]

    return train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y if split_cfg.get("stratify", True) else None
    )


def preprocess_pipeline(df: pd.DataFrame, config: dict):
    """
    Full preprocessing pipeline.

    Steps:
    - clean
    - split X/y
    - train/test split

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = clean_data(df)

    X, y = split_features_target(df, config)

    X_train, X_test, y_train, y_test = split_train_test(X, y, config)

    return X_train, X_test, y_train, y_test