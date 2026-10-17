"""Offline tests: Settings /.env isolation for the assistant flag."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from main import app


def test_settings_env_file_none_ignores_developer_dotenv(tmp_path, monkeypatch):
    """``_env_file=None`` must not load a sibling .env with ASSISTANT_ENABLED=true."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("ASSISTANT_ENABLED=true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASSISTANT_ENABLED", raising=False)

    isolated = Settings(_env_file=None)
    assert isolated.assistant_enabled is False


def test_process_env_overrides_env_file(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("ASSISTANT_ENABLED=true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSISTANT_ENABLED", "false")

    loaded = Settings(_env_file=str(dotenv))
    assert loaded.assistant_enabled is False


def test_offline_suite_forces_assistant_disabled_env():
    """conftest.pytest_configure must pin ASSISTANT_ENABLED=false for offline runs."""
    assert os.environ.get("ASSISTANT_ENABLED") == "false"


def test_lifespan_respects_isolated_disabled_settings(monkeypatch):
    isolated = Settings(_env_file=None, assistant_enabled=False)
    monkeypatch.setattr("main.settings", isolated)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/assistant/ask",
            json={"question": "What does the dirty filter sensor monitor?"},
        )
    assert response.status_code == 503
