from pydantic import BaseSettings


class Settings(BaseSettings):
    # Environment
    ENV: str = "development"

    # API
    APP_NAME: str = "Heart Failure ML API"
    VERSION: str = "1.0.0"

    # Paths
    MODEL_PATH: str = "models/trained/model_v1.pkl"
    FEATURE_PATH: str = "models/artifacts/feature_names.json"
    SHAP_BACKGROUND_PATH: str = "models/artifacts/shap_background.npy"
    SCHEMA_PATH: str = "data/validation/schema.json"

    # Logging
    LOG_CONFIG_PATH: str = "config/logging.yaml"

    class Config:
        env_file = ".env"


settings = Settings()