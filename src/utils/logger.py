import logging
import logging.config
from pathlib import Path
from src.utils.config_loader import load_yaml


def setup_logger(config_path: str = "config/logging.yaml") -> None:
    """
    Set up logging configuration from YAML file.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Logging config not found: {config_path}")

    config = load_yaml(config_path)
    logging.config.dictConfig(config)


def get_logger(name: str = "app") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name (str): Logger name

    Returns:
        logging.Logger
    """
    return logging.getLogger(name)