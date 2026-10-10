
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
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


settings = Settings()
