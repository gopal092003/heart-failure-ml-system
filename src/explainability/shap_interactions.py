import shap


def compute_shap_interactions(explainer, X):
    """
    Compute SHAP interaction values.

    Returns:
        interaction values
    """
    return explainer.shap_interaction_values(X)


def get_top_interactions(interactions, feature_names, top_n=5):
    """
    Extract strongest feature interactions.
    """
    import numpy as np

    interaction_strength = abs(interactions).mean(axis=0)

    results = []

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            results.append((
                feature_names[i],
                feature_names[j],
                interaction_strength[i][j]
            ))

    results.sort(key=lambda x: x[2], reverse=True)

    return results[:top_n]