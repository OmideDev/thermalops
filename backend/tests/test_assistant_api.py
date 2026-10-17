"""Offline FastAPI tests for POST /api/v1/assistant/ask (no Ollama/PDF/network)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.v1.assistant import get_assistant_service
from app.rag.ask import ask_with_evidence
from app.rag.llm import FakeLLMBackend
from app.rag.safety import OPERATIONAL_SAFETY_ABSTAIN_ANSWER
from app.rag.search import DeterministicHashEmbedder, build_index
from app.services.assistant import (
    AssistantBusyError,
    AssistantService,
)
from main import app

DOC_ID = "synthetic-manual.pdf"
DOC_SHA = "c" * 64
FAKE_MODEL = DeterministicHashEmbedder.model_name


def _eligible(**overrides: object) -> dict:
    base = {
        "chunk_id": "chunk-filter",
        "document_id": DOC_ID,
        "content_sha256": DOC_SHA,
        "page_number": 15,
        "chunk_index": 0,
        "section_heading": "Dirty Filter Sensor",
        "text": (
            "Dirty filter sensors monitor pressure drop across the outdoor "
            "air filters and exhaust air filters."
        ),
        "char_count": 100,
        "retrieval_eligibility": "eligible",
        "safe_for_retrieval": True,
    }
    base.update(overrides)
    return base


def _service_with_fake_llm(responses: list[str]) -> AssistantService:
    embedder = DeterministicHashEmbedder(dimension=48)
    index = build_index([_eligible()], embedder, model_name=FAKE_MODEL)
    llm = FakeLLMBackend(responses=responses)
    return AssistantService(
        index=index,
        embedder=embedder,
        llm=llm,
        document_titles={DOC_ID: "Synthetic Manual"},
        top_k=1,
        ask_fn=ask_with_evidence,
        index_build_count=1,
        _owns_llm_close=False,
    )


def _ok_payload(answer: str) -> str:
    return json.dumps(
        {
            "answer": answer,
            "evidence_ids": ["E1"],
            "insufficient_evidence": False,
        }
    )


@pytest.fixture
def client_with_assistant():
    service = _service_with_fake_llm(
        [
            _ok_payload(
                "Dirty filter sensors monitor pressure drop across the filters."
            )
        ]
    )

    def override():
        return service

    app.dependency_overrides[get_assistant_service] = override
    try:
        with TestClient(app) as client:
            yield client, service
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client_no_assistant():
    def override_unavailable():
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant is unavailable",
        )

    # Ensure dependency reports unavailable even if lifespan set something.
    app.dependency_overrides[get_assistant_service] = override_unavailable
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def test_assistant_route_exists(client_with_assistant):
    client, _service = client_with_assistant
    response = client.post(
        "/api/v1/assistant/ask",
        json={"question": "What does the dirty filter sensor monitor?"},
    )
    assert response.status_code == 200
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/api/v1/assistant/ask" in openapi.json()["paths"]


def test_valid_question_returns_answer_and_sources_schema(client_with_assistant):
    client, _service = client_with_assistant
    response = client.post(
        "/api/v1/assistant/ask",
        json={"question": "What does the dirty filter sensor monitor?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"answer", "sources"}
    assert isinstance(payload["answer"], str)
    assert isinstance(payload["sources"], list)
    assert payload["sources"]
    source = payload["sources"][0]
    assert set(source.keys()) == {"document_id", "title", "page"}
    assert source["document_id"] == DOC_ID
    assert source["title"] == "Synthetic Manual"
    assert source["page"] == 15
    assert "pressure" in payload["answer"].lower()


def test_empty_question_returns_422(client_with_assistant):
    client, _service = client_with_assistant
    response = client.post("/api/v1/assistant/ask", json={"question": ""})
    assert response.status_code == 422


def test_whitespace_question_returns_422(client_with_assistant):
    client, _service = client_with_assistant
    response = client.post("/api/v1/assistant/ask", json={"question": "   "})
    assert response.status_code == 422


def test_oversized_question_returns_422(client_with_assistant):
    client, _service = client_with_assistant
    response = client.post(
        "/api/v1/assistant/ask",
        json={"question": "a" * 2001},
    )
    assert response.status_code == 422


def test_unavailable_assistant_returns_503(client_no_assistant):
    response = client_no_assistant.post(
        "/api/v1/assistant/ask",
        json={"question": "What does the dirty filter sensor monitor?"},
    )
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_upstream_timeout_returns_504():
    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible()], embedder, model_name=FAKE_MODEL)
    llm = FakeLLMBackend(error=TimeoutError("ollama timed out"))
    service = AssistantService(
        index=index,
        embedder=embedder,
        llm=llm,
        document_titles={DOC_ID: "Synthetic Manual"},
        index_build_count=1,
    )

    app.dependency_overrides[get_assistant_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/assistant/ask",
                json={"question": "What does the dirty filter sensor monitor?"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 504
    assert "timed out" in response.json()["detail"].lower()


def test_upstream_connection_failure_returns_503():
    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible()], embedder, model_name=FAKE_MODEL)
    llm = FakeLLMBackend(error=ConnectionError("ollama down"))
    service = AssistantService(
        index=index,
        embedder=embedder,
        llm=llm,
        document_titles={DOC_ID: "Synthetic Manual"},
        index_build_count=1,
    )

    app.dependency_overrides[get_assistant_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/assistant/ask",
                json={"question": "What does the dirty filter sensor monitor?"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()
    assert "Traceback" not in response.text


def test_upstream_protocol_failure_returns_502():
    from app.rag.ollama import OllamaProtocolError

    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible()], embedder, model_name=FAKE_MODEL)
    llm = FakeLLMBackend(error=OllamaProtocolError("bad payload"))
    service = AssistantService(
        index=index,
        embedder=embedder,
        llm=llm,
        document_titles={DOC_ID: "Synthetic Manual"},
        index_build_count=1,
    )

    app.dependency_overrides[get_assistant_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/assistant/ask",
                json={"question": "What does the dirty filter sensor monitor?"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "invalid response" in response.json()["detail"].lower()


def test_busy_assistant_returns_429():
    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible()], embedder, model_name=FAKE_MODEL)
    llm = FakeLLMBackend(
        responses=[_ok_payload("should not matter while lock held")]
    )
    service = AssistantService(
        index=index,
        embedder=embedder,
        llm=llm,
        document_titles={DOC_ID: "Synthetic Manual"},
        index_build_count=1,
    )
    assert service._lock.acquire(blocking=False)

    app.dependency_overrides[get_assistant_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/assistant/ask",
                json={"question": "What does the dirty filter sensor monitor?"},
            )
    finally:
        service._lock.release()
        app.dependency_overrides.clear()

    assert response.status_code == 429
    assert "busy" in response.json()["detail"].lower()


def test_operational_abstention_returns_200_empty_sources():
    service = _service_with_fake_llm(
        [
            # Never used — pre-generation gate fires for blower corrective action.
            _ok_payload("Replace the fuse.")
        ]
    )
    app.dependency_overrides[get_assistant_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/assistant/ask",
                json={
                    "question": (
                        "What is the corrective action when a blower "
                        "fails to operate?"
                    )
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == OPERATIONAL_SAFETY_ABSTAIN_ANSWER
    assert payload["sources"] == []
    assert isinstance(service.llm, FakeLLMBackend)
    assert service.llm.calls == []


def test_informational_response_preserved(client_with_assistant):
    client, _service = client_with_assistant
    response = client.post(
        "/api/v1/assistant/ask",
        json={"question": "What does the dirty filter sensor monitor?"},
    )
    assert response.status_code == 200
    assert "Dirty filter sensors monitor pressure" in response.json()["answer"]


def test_index_not_rebuilt_per_question(client_with_assistant):
    client, service = client_with_assistant
    before = service.index_build_count
    # Reload responses for a second call.
    assert isinstance(service.llm, FakeLLMBackend)
    service.llm._responses.append(
        _ok_payload("Dirty filter sensors monitor pressure drop.")
    )
    r1 = client.post(
        "/api/v1/assistant/ask",
        json={"question": "What does the dirty filter sensor monitor?"},
    )
    r2 = client.post(
        "/api/v1/assistant/ask",
        json={"question": "What does the dirty filter sensor monitor?"},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert service.index_build_count == before == 1


def test_equipment_still_works_when_assistant_unavailable(client_no_assistant):
    # Health does not need DB; equipment list needs DB override like other tests.
    response = client_no_assistant.get("/health")
    assert response.status_code == 200

    # Assistant remains 503
    ask = client_no_assistant.post(
        "/api/v1/assistant/ask",
        json={"question": "What does the dirty filter sensor monitor?"},
    )
    assert ask.status_code == 503


def test_default_app_assistant_disabled_without_override(monkeypatch):
    """
    Lifespan with assistant disabled leaves the route unavailable.

    Uses an isolated Settings object (no .env) so a developer machine with
    ASSISTANT_ENABLED=true in an ignored .env cannot flip this offline test.
    """
    from app.core.config import Settings

    isolated = Settings(_env_file=None, assistant_enabled=False)
    monkeypatch.setattr("main.settings", isolated)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/assistant/ask",
            json={"question": "What does the dirty filter sensor monitor?"},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "Assistant is unavailable"


def test_service_busy_error_helper():
    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible()], embedder, model_name=FAKE_MODEL)
    service = AssistantService(
        index=index,
        embedder=embedder,
        llm=FakeLLMBackend(responses=[_ok_payload("x")]),
        index_build_count=1,
    )
    service._lock.acquire()
    try:
        with pytest.raises(AssistantBusyError):
            service.ask("What does the dirty filter sensor monitor?")
    finally:
        service._lock.release()


def test_lock_released_after_upstream_failure():
    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible()], embedder, model_name=FAKE_MODEL)
    llm = FakeLLMBackend(error=TimeoutError("boom"))
    service = AssistantService(
        index=index,
        embedder=embedder,
        llm=llm,
        document_titles={DOC_ID: "Synthetic Manual"},
        index_build_count=1,
    )
    with pytest.raises(TimeoutError):
        service.ask("What does the dirty filter sensor monitor?")
    # Lock must be free for the next caller.
    assert service._lock.acquire(blocking=False)
    service._lock.release()


def test_build_from_manual_closes_owned_llm_on_late_failure(monkeypatch):
    from app.services import assistant as assistant_mod

    class BoomBackend:
        def __init__(self, **_kwargs):
            self.closed = False

        def close(self) -> None:
            self.closed = True

    created: list[BoomBackend] = []

    def fake_ollama(**_kwargs):
        backend = BoomBackend()
        created.append(backend)
        return backend

    monkeypatch.setattr(assistant_mod, "OllamaChatBackend", fake_ollama)
    monkeypatch.setattr(
        assistant_mod,
        "load_eligible_chunks_from_manual",
        lambda *_args, **_kwargs: (
            [_eligible()],
            {"document_id": DOC_ID},
        ),
    )
    monkeypatch.setattr(
        assistant_mod,
        "build_index",
        lambda *_args, **_kwargs: MagicMock(size=1),
    )

    def boom_init(self, *args, **kwargs):
        raise RuntimeError("simulate service construction failure")

    monkeypatch.setattr(AssistantService, "__init__", boom_init)

    embedder = DeterministicHashEmbedder(dimension=32)
    with pytest.raises(RuntimeError, match="simulate service"):
        AssistantService.build_from_manual(
            Path("unused.pdf"),
            llm=None,
            embedder=embedder,
        )
    assert created
    assert created[0].closed is True
