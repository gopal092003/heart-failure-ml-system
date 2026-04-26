import numpy as np


def validate_input_dict(input_data: dict, feature_order: list):
    """
    Validate API input dictionary.
    """
    missing = set(feature_order) - set(input_data.keys())
    extra = set(input_data.keys()) - set(feature_order)

    if missing:
        raise ValueError(f"Missing features: {missing}")

    if extra:
        raise ValueError(f"Unexpected features: {extra}")


def validate_no_nan(X: np.ndarray):
    """
    Ensure no NaN values.
    """
    if np.isnan(X).any():
        raise ValueError("Input contains NaN values")


def validate_ranges(input_data: dict, schema: dict):
    """
    Validate input ranges using schema.
    """
    for feature, value in input_data.items():
        if feature not in schema["columns"]:
            continue

        rules = schema["columns"][feature]

        if "min" in rules and value < rules["min"]:
            raise ValueError(f"{feature} below minimum {rules['min']}")

        if "max" in rules and value > rules["max"]:
            raise ValueError(f"{feature} above maximum {rules['max']}")

        if "allowed_values" in rules and value not in rules["allowed_values"]:
            raise ValueError(f"{feature} has invalid value {value}")


def validate_input_array(X: np.ndarray, expected_dim: int):
    """
    Validate input array shape.
    """
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} features, got {X.shape[1]}"
        )


def full_input_validation(input_data: dict, feature_order: list, schema: dict):
    """
    Complete validation pipeline.
    """
    validate_input_dict(input_data, feature_order)
    validate_ranges(input_data, schema)