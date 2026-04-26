from src.features.build import build_preprocessor
from src.features.feature_checks import check_feature_consistency


def test_preprocessor_build():
    config = {
        "features": {
            "numeric": ["age", "platelets"],
            "categorical": ["sex"]
        }
    }

    preprocessor = build_preprocessor(config)

    assert preprocessor is not None


def test_feature_consistency():
    import pandas as pd

    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

    # Should not raise error
    check_feature_consistency(df1, df2)