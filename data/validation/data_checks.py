import json
import pandas as pd


def load_schema(schema_path: str) -> dict:
    with open(schema_path, "r") as f:
        return json.load(f)


def check_required_columns(df: pd.DataFrame, schema: dict):
    missing_cols = set(schema["required_columns"]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def check_column_types(df: pd.DataFrame, schema: dict):
    for col, rules in schema["columns"].items():
        if col not in df.columns:
            continue

        expected_type = rules["type"]

        if expected_type == "int":
            if not pd.api.types.is_integer_dtype(df[col]):
                raise TypeError(f"Column '{col}' should be int")

        elif expected_type == "float":
            if not pd.api.types.is_float_dtype(df[col]) and not pd.api.types.is_integer_dtype(df[col]):
                raise TypeError(f"Column '{col}' should be float")


def check_missing_values(df: pd.DataFrame):
    if df.isnull().sum().sum() > 0:
        missing = df.isnull().sum()
        raise ValueError(f"Missing values found:\n{missing}")


def check_value_ranges(df: pd.DataFrame, schema: dict):
    for col, rules in schema["columns"].items():
        if col not in df.columns:
            continue

        if "min" in rules:
            if (df[col] < rules["min"]).any():
                raise ValueError(f"Column '{col}' has values below minimum {rules['min']}")

        if "max" in rules:
            if (df[col] > rules["max"]).any():
                raise ValueError(f"Column '{col}' has values above maximum {rules['max']}")


def check_allowed_values(df: pd.DataFrame, schema: dict):
    for col, rules in schema["columns"].items():
        if col not in df.columns:
            continue

        if "allowed_values" in rules:
            invalid = ~df[col].isin(rules["allowed_values"])
            if invalid.any():
                raise ValueError(
                    f"Column '{col}' contains invalid values: {df[col][invalid].unique()}"
                )


def validate_dataframe(df: pd.DataFrame, schema_path: str):
    schema = load_schema(schema_path)

    print("🔍 Running data validation checks...")

    check_required_columns(df, schema)
    check_column_types(df, schema)
    check_missing_values(df)
    check_value_ranges(df, schema)
    check_allowed_values(df, schema)

    print("✅ Data validation passed successfully!")


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/raw/heart_failure_clinical_records_dataset.csv")
    validate_dataframe(df, "data/validation/schema.json")