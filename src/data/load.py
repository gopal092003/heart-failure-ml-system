import pandas as pd
from pathlib import Path


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset.

    Args:
        path (str): Path to raw CSV file

    Returns:
        pd.DataFrame
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Raw data not found at: {path}")

    df = pd.read_csv(file_path)
    return df


def load_processed_data(path: str) -> pd.DataFrame:
    """
    Load processed dataset (parquet).

    Args:
        path (str): Path to parquet file

    Returns:
        pd.DataFrame
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Processed data not found at: {path}")

    df = pd.read_parquet(file_path)
    return df


def save_processed_data(df: pd.DataFrame, path: str):
    """
    Save processed data to parquet.

    Args:
        df (pd.DataFrame)
        path (str)
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(file_path, index=False)