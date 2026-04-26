from src.utils.config_loader import load_config


def get_config():
    """
    Load main project config.
    """
    return load_config("config/config.yaml")