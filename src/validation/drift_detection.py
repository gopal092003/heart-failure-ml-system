import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def ks_drift_test(train_series, new_series, alpha=0.05):
    """
    Kolmogorov-Smirnov test for drift.
    """
    stat, p_value = ks_2samp(train_series, new_series)

    drift_detected = p_value < alpha

    return {
        "statistic": stat,
        "p_value": p_value,
        "drift": drift_detected
    }


def detect_drift(train_df: pd.DataFrame, new_df: pd.DataFrame, numeric_features: list):
    """
    Detect drift across multiple features.
    """
    results = {}

    for col in numeric_features:
        if col not in train_df.columns or col not in new_df.columns:
            continue

        result = ks_drift_test(train_df[col], new_df[col])
        results[col] = result

    return results


def summarize_drift(drift_results: dict):
    """
    Summarize drift detection.
    """
    drifted_features = [k for k, v in drift_results.items() if v["drift"]]

    summary = {
        "total_features": len(drift_results),
        "drifted_features": drifted_features,
        "num_drifted": len(drifted_features)
    }

    return summary


def log_drift_results(drift_results: dict):
    """
    Print drift results (can be replaced with logging).
    """
    print("📊 Drift Detection Results:")

    for feature, result in drift_results.items():
        status = "⚠️ Drift" if result["drift"] else "✅ Stable"
        print(f"{feature}: p={result['p_value']:.4f} → {status}")