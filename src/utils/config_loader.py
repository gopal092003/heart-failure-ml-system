import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to config file

    Returns:
        dict: Parsed configuration
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_yaml(path: str) -> dict:
    """
    Generic YAML loader (for logging.yaml, model_registry.yaml, etc.)
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(file_path, "r") as f:
        return yaml.safe_load(f)