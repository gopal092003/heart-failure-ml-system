import pandas as pd
from src.data.load import load_raw_data
from src.data.preprocess import split_features_target


def test_load_data():
    df = load_raw_data("data/raw/heart_failure_clinical_records_dataset.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_split_features_target():
    df = load_raw_data("data/raw/heart_failure_clinical_records_dataset.csv")

    config = {
        "data": {
            "target_column": "DEATH_EVENT",
            "drop_columns": ["time"]
        }
    }

    X, y = split_features_target(df, config)

    assert "DEATH_EVENT" not in X.columns
    assert len(X) == len(y)