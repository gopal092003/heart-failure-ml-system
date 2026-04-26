import pandas as pd


def check_feature_consistency(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Ensure train and test have same columns.
    """
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)

    if train_cols != test_cols:
        raise ValueError(
            f"Mismatch in train/test columns:\n"
            f"Only in train: {train_cols - test_cols}\n"
            f"Only in test: {test_cols - train_cols}"
        )


def check_no_constant_features(df: pd.DataFrame):
    """
    Detect constant columns.
    """
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

    if constant_cols:
        print(f"⚠️ Constant features detected: {constant_cols}")


def check_feature_ranges(df: pd.DataFrame):
    """
    Basic sanity check for feature distributions.
    """
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            if df[col].min() == df[col].max():
                print(f"⚠️ Feature '{col}' has no variation")


def check_data_leakage(df: pd.DataFrame, target_column: str):
    """
    Check if any feature is perfectly correlated with target.
    """
    if target_column not in df.columns:
        return

    correlations = df.corr(numeric_only=True)[target_column].drop(target_column)

    suspicious = correlations[abs(correlations) > 0.95]

    if not suspicious.empty:
        print("⚠️ Potential data leakage detected:")
        print(suspicious)


def run_feature_checks(X_train: pd.DataFrame, X_test: pd.DataFrame, df_full: pd.DataFrame, config: dict):
    """
    Run all feature checks.
    """
    print("🔍 Running feature checks...")

    check_feature_consistency(X_train, X_test)
    check_no_constant_features(X_train)
    check_feature_ranges(X_train)

    target = config["data"]["target_column"]
    check_data_leakage(df_full, target)

    print("✅ Feature checks completed")