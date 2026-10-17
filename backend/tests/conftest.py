"""Pytest configuration for ThermalOps backend tests."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live_ollama: requires RUN_OLLAMA_LIVE=1 and local Ollama + qwen3:4b-instruct",
    )
    config.addinivalue_line(
        "markers",
        "assistant_e2e: requires RUN_ASSISTANT_E2E=1, local PDF, FastEmbed, Ollama",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip opt-in live tests unless their env flags are set."""
    skip_ollama = pytest.mark.skip(
        reason="live Ollama tests require RUN_OLLAMA_LIVE=1"
    )
    skip_e2e = pytest.mark.skip(
        reason="assistant API E2E requires RUN_ASSISTANT_E2E=1"
    )
    run_ollama = os.environ.get("RUN_OLLAMA_LIVE") == "1"
    run_e2e = os.environ.get("RUN_ASSISTANT_E2E") == "1"
    for item in items:
        if "live_ollama" in item.keywords and not run_ollama:
            item.add_marker(skip_ollama)
        if "assistant_e2e" in item.keywords and not run_e2e:
            item.add_marker(skip_e2e)
