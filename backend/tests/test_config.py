
from app.core.config import Settings


def test_settings_use_default_values():
    settings = Settings(_env_file=None)

    assert settings.app_name == "ThermalOps API"
    assert settings.app_version == "0.1.0"


def test_settings_accept_custom_values():
    settings = Settings(
        app_name="ThermalOps Test",
        app_version="0.2.0",
        _env_file=None,
    )

    assert settings.app_name == "ThermalOps Test"
    assert settings.app_version == "0.2.0"
