"""Offline tests for assistant request timing helpers and wiring."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.rag.ask import ask_with_evidence
from app.rag.llm import FakeLLMBackend
from app.rag.ollama import OllamaChatBackend
from app.rag.search import DeterministicHashEmbedder, build_index
from app.rag.timing import (
    AssistantRequestMetrics,
    extract_ollama_timing,
    ns_to_ms,
    tokens_per_second,
)
from app.services.assistant import AssistantService


DOC_ID = "synthetic-manual.pdf"
DOC_SHA = "d" * 64


def _eligible() -> dict:
    return {
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


def _ok_llm() -> str:
    return json.dumps(
        {
            "answer": "Monitors pressure drop across the filters.",
            "evidence_ids": ["E1"],
            "insufficient_evidence": False,
        }
    )


def test_ns_to_ms_and_missing():
    assert ns_to_ms(1_000_000_000) == pytest.approx(1000.0)
    assert ns_to_ms(None) is None
    assert ns_to_ms("slow") is None
    assert ns_to_ms(True) is None


def test_tokens_per_second_requires_positive_duration():
    assert tokens_per_second(eval_count=50, eval_duration_ns=5_000_000_000) == pytest.approx(
        10.0
    )
    assert tokens_per_second(eval_count=50, eval_duration_ns=0) is None
    assert tokens_per_second(eval_count=50, eval_duration_ns=-1) is None
    assert tokens_per_second(eval_count="x", eval_duration_ns=1_000_000_000) is None


def test_extract_ollama_timing_present_and_absent():
    full = extract_ollama_timing(
        {
            "total_duration": 60_000_000_000,
            "load_duration": 2_000_000_000,
            "prompt_eval_count": 120,
            "prompt_eval_duration": 3_000_000_000,
            "eval_count": 40,
            "eval_duration": 10_000_000_000,
            "done_reason": "stop",
        }
    )
    assert full.total_duration_ms == pytest.approx(60_000.0)
    assert full.load_duration_ms == pytest.approx(2_000.0)
    assert full.prompt_eval_count == 120
    assert full.prompt_eval_duration_ms == pytest.approx(3_000.0)
    assert full.eval_count == 40
    assert full.eval_duration_ms == pytest.approx(10_000.0)
    assert full.done_reason == "stop"
    assert full.eval_tokens_per_sec == pytest.approx(4.0)

    empty = extract_ollama_timing({"message": {"content": "{}"}})
    assert empty.total_duration_ms is None
    assert empty.eval_count is None
    assert empty.eval_tokens_per_sec is None
    assert empty.done_reason is None


def test_ollama_backend_captures_timing_from_mocked_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": _ok_llm(),
                },
                "done": True,
                "done_reason": "stop",
                "total_duration": 5_000_000_000,
                "load_duration": 500_000_000,
                "prompt_eval_count": 10,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 20,
                "eval_duration": 2_000_000_000,
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://127.0.0.1:11434")
    backend = OllamaChatBackend(http_client=client)
    try:
        content = backend.complete(system="s", user="u")
    finally:
        backend.close()
        client.close()

    assert json.loads(content)["evidence_ids"] == ["E1"]
    assert backend.last_timing is not None
    assert backend.last_timing.load_duration_ms == pytest.approx(500.0)
    assert backend.last_timing.eval_tokens_per_sec == pytest.approx(10.0)


def test_ollama_last_timing_cleared_before_failed_generation():
    """A transport failure must not leave prior timing attached."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": _ok_llm()},
                    "done_reason": "stop",
                    "eval_count": 20,
                    "eval_duration": 2_000_000_000,
                },
            )
        raise httpx.ReadTimeout("slow", request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://127.0.0.1:11434")
    backend = OllamaChatBackend(http_client=client, read_timeout=0.1)
    try:
        backend.complete(system="s", user="u")
        assert backend.last_timing is not None
        assert backend.last_timing.eval_count == 20
        with pytest.raises(TimeoutError):
            backend.complete(system="s", user="u")
        assert backend.last_timing is None
        assert backend.last_raw_response is None
    finally:
        backend.close()
        client.close()


def test_failed_llm_call_does_not_copy_stale_ollama_timing_into_metrics():
    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible()], embedder, model_name=embedder.model_name)

    class StaleThenFailLLM:
        def __init__(self) -> None:
            from app.rag.timing import OllamaTiming

            self.last_timing = OllamaTiming(eval_count=99, eval_duration_ms=123.0)

        def complete(self, *, system: str, user: str) -> str:
            self.last_timing = None
            raise TimeoutError("upstream timed out")

    metrics = AssistantRequestMetrics()
    with pytest.raises(TimeoutError):
        ask_with_evidence(
            "What does the dirty filter sensor monitor?",
            index,
            embedder,
            StaleThenFailLLM(),  # type: ignore[arg-type]
            top_k=1,
            metrics=metrics,
        )

    assert metrics.llm_wall_ms is not None and metrics.llm_wall_ms >= 0
    assert metrics.ollama.eval_count is None
    assert metrics.retrieval_ms is not None


def test_ask_with_evidence_fills_metrics_without_changing_schema():
    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible()], embedder, model_name=embedder.model_name)
    llm = FakeLLMBackend(responses=[_ok_llm()])
    metrics = AssistantRequestMetrics()

    result = ask_with_evidence(
        "What does the dirty filter sensor monitor?",
        index,
        embedder,
        llm,
        top_k=1,
        document_titles={DOC_ID: "Synthetic"},
        metrics=metrics,
    )

    assert set(result.keys()) == {"answer", "sources"}
    assert result["sources"][0]["page"] == 15
    assert metrics.outcome == "answer"
    assert metrics.retrieval_ms is not None and metrics.retrieval_ms >= 0
    assert metrics.llm_wall_ms is not None and metrics.llm_wall_ms >= 0
    # Fake LLM has no Ollama timing fields.
    assert metrics.ollama.eval_count is None


def test_operational_abstention_metrics_skip_llm():
    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible()], embedder, model_name=embedder.model_name)
    llm = FakeLLMBackend(responses=[_ok_llm()])
    metrics = AssistantRequestMetrics()

    result = ask_with_evidence(
        "What corrective action should I take if the blower is not working?",
        index,
        embedder,
        llm,
        metrics=metrics,
    )

    assert result["sources"] == []
    assert metrics.outcome == "abstain_operational"
    assert metrics.retrieval_ms is None
    assert metrics.llm_wall_ms is None
    assert llm.calls == []


def test_service_ask_logs_metrics_and_keeps_response_contract(caplog):
    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible()], embedder, model_name=embedder.model_name)
    service = AssistantService(
        index=index,
        embedder=embedder,
        llm=FakeLLMBackend(responses=[_ok_llm()]),
        document_titles={DOC_ID: "Synthetic"},
        top_k=1,
    )

    with caplog.at_level(logging.INFO, logger="app.rag.timing"):
        result = service.ask("What does the dirty filter sensor monitor?")

    assert set(result.keys()) == {"answer", "sources"}
    assert any("assistant_request" in rec.message for rec in caplog.records)
    # Must not leak the question text into the timing log line.
    for rec in caplog.records:
        if "assistant_request" in rec.message:
            assert "dirty filter" not in rec.message.lower()
