"""Offline unit tests for experimental OllamaChatBackend (no live Ollama)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.rag.ollama import (
    DEFAULT_OLLAMA_MODEL,
    PHASE_4A_RESPONSE_JSON_SCHEMA,
    OllamaChatBackend,
    OllamaProtocolError,
    OllamaUnavailableError,
)


def _ok_chat_payload(answer: str = "ok") -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "answer": answer,
                    "evidence_ids": ["E1"],
                    "insufficient_evidence": False,
                }
            ),
        },
        "done": True,
        "done_reason": "stop",
    }


def test_complete_posts_localhost_schema_and_returns_content_only():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        body = json.loads(request.content.decode("utf-8"))
        captured["body"] = body
        return httpx.Response(200, json=_ok_chat_payload("sensor monitors pressure"))

    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    )
    backend = OllamaChatBackend(http_client=client, model=DEFAULT_OLLAMA_MODEL)
    try:
        content = backend.complete(system="sys", user="usr")
    finally:
        backend.close()
        client.close()

    assert "sensor monitors pressure" in content
    parsed = json.loads(content)
    assert parsed["evidence_ids"] == ["E1"]
    assert captured["url"].endswith("/api/chat")
    body = captured["body"]
    assert body["model"] == "qwen3:4b-instruct"
    assert body["stream"] is False
    assert body["format"] == PHASE_4A_RESPONSE_JSON_SCHEMA
    assert body["options"]["num_gpu"] == 0
    assert body["options"]["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1] == {"role": "user", "content": "usr"}


def test_rejects_non_localhost_base_url():
    with pytest.raises(ValueError, match="localhost"):
        OllamaChatBackend(base_url="http://example.com:11434")


def test_timeout_maps_to_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    )
    backend = OllamaChatBackend(http_client=client, read_timeout=0.1)
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            backend.complete(system="s", user="u")
    finally:
        backend.close()
        client.close()


def test_connection_failure_maps_to_unavailable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    )
    backend = OllamaChatBackend(http_client=client)
    try:
        with pytest.raises(OllamaUnavailableError, match="unavailable"):
            backend.complete(system="s", user="u")
    finally:
        backend.close()
        client.close()


def test_http_error_status_raises_protocol_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        transport=transport,
        base_url="http://127.0.0.1:11434",
    )
    backend = OllamaChatBackend(http_client=client)
    try:
        with pytest.raises(OllamaProtocolError, match="unexpected HTTP status"):
            backend.complete(system="s", user="u")
    finally:
        backend.close()
        client.close()
