"""Pytest configuration for ThermalOps backend tests."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live_ollama: requires RUN_OLLAMA_LIVE=1 and local Ollama + qwen3:4b-instruct",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip live Ollama tests unless explicitly enabled."""
    if os.environ.get("RUN_OLLAMA_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(
        reason="live Ollama tests require RUN_OLLAMA_LIVE=1"
    )
    for item in items:
        if "live_ollama" in item.keywords:
            item.add_marker(skip_live)
