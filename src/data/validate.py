import pandas as pd
from pathlib import Path
from data.validation.data_checks import validate_dataframe


def validate_data(df: pd.DataFrame, schema_path: str):
    """
    Validate dataframe against schema.

    Args:
        df (pd.DataFrame)
        schema_path (str)
    """
    schema_file = Path(schema_path)

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    validate_dataframe(df, schema_path)


def validate_raw_data(df: pd.DataFrame):
    """
    Validate raw dataset using default schema.
    """
    validate_data(df, "data/validation/schema.json")