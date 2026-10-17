from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.

    Production loads optional ``.env`` plus process environment. Tests that
    must ignore developer ``.env`` should construct
    ``Settings(_env_file=None, ...)`` or rely on pytest forcing
    ``ASSISTANT_ENABLED=false`` (env wins over env_file).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "ThermalOps API"
    app_version: str = "0.1.0"

    database_host: str = "127.0.0.1"
    database_port: int = 5432
    database_name: str = "thermalops"
    database_user: str = "thermalops_app"
    database_password: str | None = Field(default=None, repr=False)

    # Experimental grounded assistant (disabled by default for CI / offline).
    assistant_enabled: bool = False
    assistant_manual_path: str | None = None
    assistant_allowed_pages: str = "15,29,33"
    assistant_document_title: str | None = "Energy Recovery Ventilator IOM"
    assistant_ollama_base_url: str = "http://127.0.0.1:11434"
    assistant_ollama_model: str = "qwen3:4b-instruct"
    assistant_ollama_connect_timeout: float = 5.0
    # Local CPU generation can exceed 90s; keep frontend cancel separate.
    assistant_ollama_read_timeout: float = 180.0
    assistant_top_k: int = 3


settings = Settings()
