"""
Experimental local Ollama chat backend (Phase 4B.3).

Talks only to a localhost Ollama HTTP API. Not production deployment.
JSON Schema structured output matches the Phase 4A ask contract; strict
parsing remains in ask_with_evidence.

``complete`` uses blocking httpx HTTP. When wiring FastAPI later, call it from
a worker/thread or replace with an async client so the event loop is not blocked.
No cloud fallback and no API credentials are used or logged.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from app.rag.timing import OllamaTiming, extract_ollama_timing

# Exact Phase 4A structured-output schema (object form for Ollama `format`).
PHASE_4A_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": ["answer", "evidence_ids", "insufficient_evidence"],
    "additionalProperties": False,
}

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct"


class OllamaUnavailableError(ConnectionError):
    """Ollama daemon unreachable or connection failed (map to HTTP 503)."""


class OllamaProtocolError(RuntimeError):
    """Invalid Ollama HTTP status or response body (map to HTTP 502)."""


def _assert_localhost_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            "OllamaChatBackend only allows localhost base URLs "
            f"(got {base_url!r}); refusing non-local endpoints"
        )
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported Ollama URL scheme: {base_url!r}")
    return base_url.rstrip("/")


class OllamaChatBackend:
    """
    LLMBackend implementation for local Ollama /api/chat.

    Inject ``http_client`` (httpx.Client) for offline unit tests. No cloud
    fallback and no API credentials.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        connect_timeout: float = 5.0,
        read_timeout: float = 180.0,
        temperature: float = 0.0,
        num_ctx: int = 2048,
        num_predict: int = 192,
        num_gpu: int = 0,
        http_client: httpx.Client | None = None,
    ) -> None:
        if connect_timeout <= 0 or read_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if num_ctx < 1 or num_predict < 1:
            raise ValueError("num_ctx and num_predict must be >= 1")
        if not model or not str(model).strip():
            raise ValueError("model must be a non-empty string")

        self.base_url = _assert_localhost_base_url(base_url)
        self.model = str(model).strip()
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self.temperature = float(temperature)
        self.num_ctx = int(num_ctx)
        self.num_predict = int(num_predict)
        self.num_gpu = int(num_gpu)
        self._client = http_client
        self._owns_client = http_client is None
        self.last_request: dict[str, Any] | None = None
        self.last_raw_response: dict[str, Any] | None = None
        self.last_timing: OllamaTiming | None = None

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            timeout = httpx.Timeout(
                connect=self.connect_timeout,
                read=self.read_timeout,
                write=self.read_timeout,
                pool=self.connect_timeout,
            )
            self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
            self._owns_client = True
        return self._client

    def complete(self, *, system: str, user: str) -> str:
        """Return assistant message content only (JSON string expected)."""
        if not isinstance(system, str) or not isinstance(user, str):
            raise TypeError("system and user must be strings")

        # Drop prior request artifacts so a failed generation cannot leak
        # timing into the next request's metrics snapshot.
        self.last_request = None
        self.last_raw_response = None
        self.last_timing = None

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": PHASE_4A_RESPONSE_JSON_SCHEMA,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "num_gpu": self.num_gpu,
            },
        }
        self.last_request = payload

        client = self._get_client()
        try:
            response = client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                "Ollama request timed out talking to localhost"
            ) from exc
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                "Ollama service is unavailable on localhost"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaProtocolError(
                "Ollama returned an unexpected HTTP status"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(
                "Ollama HTTP transport failed on localhost"
            ) from exc

        if not isinstance(data, dict):
            raise OllamaProtocolError("Ollama response is not a JSON object")
        self.last_raw_response = data
        self.last_timing = extract_ollama_timing(data)

        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaProtocolError("Ollama response missing message object")
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaProtocolError("Ollama message content must be a string")
        return content
