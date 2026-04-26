import logging
import logging.config
from src.utils.config_loader import load_yaml
from app.core.settings import settings


def setup_api_logger():
    """
    Setup logging for FastAPI.
    """
    config = load_yaml(settings.LOG_CONFIG_PATH)
    logging.config.dictConfig(config)


def get_logger(name: str = "api") -> logging.Logger:
    return logging.getLogger(name)