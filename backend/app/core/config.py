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

    # Phase 6B.1 admin auth — never expose these via Vite env.
    # Fail closed when username/hash missing (admin routes unavailable).
    admin_username: str | None = Field(default=None, repr=False)
    admin_password_hash: str | None = Field(
        default=None,
        repr=False,
        description="Argon2id encoded hash (argon2.PasswordHasher().hash(...))",
    )
    admin_session_ttl_seconds: int = 60 * 60 * 8
    admin_cookie_name: str = "thermalops_admin_session"
    admin_cookie_secure: bool = False  # True for HTTPS staging/production
    admin_cookie_samesite: str = "lax"
    admin_csrf_header_name: str = "X-CSRF-Token"
    # Comma-separated Origins allowed for cookie-authenticated admin mutations.
    # Must stay aligned with CORS allow_origins for the Vite app.
    admin_allowed_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173"
    )
    admin_login_max_failures: int = 5
    admin_login_window_seconds: int = 300

    # RAG document management feature flag (persistence only in 6B.1).
    rag_documents_enabled: bool = False
    rag_storage_root: str = "data/rag_private"


settings = Settings()


def admin_auth_configured(cfg: Settings | None = None) -> bool:
    """Return True when admin username and Argon2id hash are both set."""
    current = cfg if cfg is not None else settings
    username = (current.admin_username or "").strip()
    password_hash = (current.admin_password_hash or "").strip()
    return bool(username and password_hash)
