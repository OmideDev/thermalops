"""
Optional live end-to-end assistant API test.

Requires:
  RUN_ASSISTANT_E2E=1
  ASSISTANT_ENABLED path prerequisites (manual PDF + FastEmbed + Ollama qwen3:4b-instruct)

Not part of default CI.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.rag.ollama import DEFAULT_OLLAMA_MODEL
from app.rag.safety import OPERATIONAL_SAFETY_ABSTAIN_ANSWER
from app.services.assistant import (
    BACKEND_ROOT,
    try_create_assistant_from_settings,
)
from main import app

pytestmark = pytest.mark.assistant_e2e


def _prereqs() -> tuple[bool, str]:
    manual = settings.assistant_manual_path
    path = (
        Path(manual)
        if manual
        else BACKEND_ROOT / "data" / "manuals" / "460945ERT_iom.pdf"
    )
    if not path.is_absolute():
        path = (BACKEND_ROOT / path).resolve()
    if not path.is_file():
        return False, f"manual missing: {path.name}"
    try:
        tags = httpx.get("http://127.0.0.1:11434/api/tags", timeout=3.0)
        tags.raise_for_status()
        names = {m.get("name") for m in tags.json().get("models", [])}
        if DEFAULT_OLLAMA_MODEL not in names and (
            f"{DEFAULT_OLLAMA_MODEL}:latest" not in names
        ):
            return False, f"model {DEFAULT_OLLAMA_MODEL} not installed"
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama unreachable: {exc}"
    return True, str(path)


@pytest.fixture(scope="module")
def live_client():
    if os.environ.get("RUN_ASSISTANT_E2E") != "1":
        pytest.skip("set RUN_ASSISTANT_E2E=1 for live assistant API E2E")
    ok, detail = _prereqs()
    if not ok:
        pytest.fail(f"RUN_ASSISTANT_E2E=1 but prerequisites failed ({detail})")

    settings.assistant_enabled = True
    service = try_create_assistant_from_settings(settings)
    if service is None:
        pytest.fail("assistant failed to initialize for live E2E")

    with TestClient(app) as client:
        app.state.assistant = service
        yield client, service
        service.close()
        app.state.assistant = None


def test_live_e2e_dirty_filter_frost_and_blower(live_client):
    client, service = live_client
    print(f"\nE2E index_size={service.index.size} builds={service.index_build_count}")

    cases = [
        ("What does the dirty filter sensor monitor?", False),
        ("How does frost control work?", False),
        (
            "What is the corrective action when a blower fails to operate?",
            True,
        ),
    ]
    for question, expect_ops_abstain in cases:
        started = time.perf_counter()
        response = client.post(
            "/api/v1/assistant/ask",
            json={"question": question},
        )
        elapsed = time.perf_counter() - started
        print(f"\nE2E question={question!r}")
        print(f"E2E status={response.status_code} latency_s={elapsed:.3f}")
        print(f"E2E body={response.json()}")
        assert response.status_code == 200
        body = response.json()
        assert "answer" in body and "sources" in body
        if expect_ops_abstain:
            assert body["answer"] == OPERATIONAL_SAFETY_ABSTAIN_ANSWER
            assert body["sources"] == []
        else:
            assert body["sources"], "expected grounded sources"
            for source in body["sources"]:
                assert source["page"] in {15, 29, 33}
                assert source["page"] != 31
                assert source["document_id"]
                assert source["title"]
    assert service.index_build_count == 1
