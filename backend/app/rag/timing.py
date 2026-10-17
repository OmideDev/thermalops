"""
Request-scoped timing helpers for the experimental grounded assistant.

Wall times use ``time.perf_counter``. Ollama fields are copied only when present
in the upstream JSON (nanoseconds → milliseconds). No process-global mutable
timing state — callers pass an ``AssistantRequestMetrics`` instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def ns_to_ms(value: Any) -> float | None:
    """Convert Ollama nanosecond durations to milliseconds when numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) / 1_000_000.0
    return None


def tokens_per_second(*, eval_count: Any, eval_duration_ns: Any) -> float | None:
    """
    Output tokens/sec from Ollama eval_* fields.

    Returns None unless ``eval_count`` is a non-negative number and
    ``eval_duration_ns`` is a positive number.
    """
    if isinstance(eval_count, bool) or isinstance(eval_duration_ns, bool):
        return None
    if not isinstance(eval_count, (int, float)):
        return None
    if not isinstance(eval_duration_ns, (int, float)):
        return None
    if eval_count < 0 or eval_duration_ns <= 0:
        return None
    seconds = float(eval_duration_ns) / 1_000_000_000.0
    if seconds <= 0:
        return None
    return float(eval_count) / seconds


@dataclass
class OllamaTiming:
    """Subset of Ollama /api/chat timing fields (absent fields stay None)."""

    total_duration_ms: float | None = None
    load_duration_ms: float | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ms: float | None = None
    eval_count: int | None = None
    eval_duration_ms: float | None = None
    done_reason: str | None = None
    eval_tokens_per_sec: float | None = None


def extract_ollama_timing(payload: dict[str, Any] | None) -> OllamaTiming:
    """Parse timing fields from an Ollama chat response object; never invent."""
    out = OllamaTiming()
    if not isinstance(payload, dict):
        return out

    out.total_duration_ms = ns_to_ms(payload.get("total_duration"))
    out.load_duration_ms = ns_to_ms(payload.get("load_duration"))
    out.prompt_eval_duration_ms = ns_to_ms(payload.get("prompt_eval_duration"))
    out.eval_duration_ms = ns_to_ms(payload.get("eval_duration"))

    pec = payload.get("prompt_eval_count")
    if isinstance(pec, int) and not isinstance(pec, bool):
        out.prompt_eval_count = pec

    ec = payload.get("eval_count")
    if isinstance(ec, int) and not isinstance(ec, bool):
        out.eval_count = ec

    reason = payload.get("done_reason")
    if isinstance(reason, str) and reason.strip():
        out.done_reason = reason.strip()

    out.eval_tokens_per_sec = tokens_per_second(
        eval_count=payload.get("eval_count"),
        eval_duration_ns=payload.get("eval_duration"),
    )
    return out


@dataclass
class AssistantRequestMetrics:
    """
    Per-request timings for one assistant generation.

    Owned by ``AssistantService.ask`` for the duration of the lock. Nested
    helpers may fill fields but must not share instances across requests.
    """

    prepare_ms: float | None = None
    retrieval_ms: float | None = None
    llm_wall_ms: float | None = None
    total_ms: float | None = None
    ollama: OllamaTiming = field(default_factory=OllamaTiming)
    outcome: str = "unknown"  # answer | abstain_operational | abstain_evidence | error


def log_assistant_request_metrics(metrics: AssistantRequestMetrics) -> None:
    """Structured info log without question text, prompts, or paths."""
    ollama = metrics.ollama
    logger.info(
        "assistant_request outcome=%s total_ms=%s prepare_ms=%s retrieval_ms=%s "
        "llm_wall_ms=%s ollama_total_ms=%s ollama_load_ms=%s "
        "prompt_eval_count=%s prompt_eval_ms=%s eval_count=%s eval_ms=%s "
        "eval_tok_s=%s done_reason=%s",
        metrics.outcome,
        _fmt(metrics.total_ms),
        _fmt(metrics.prepare_ms),
        _fmt(metrics.retrieval_ms),
        _fmt(metrics.llm_wall_ms),
        _fmt(ollama.total_duration_ms),
        _fmt(ollama.load_duration_ms),
        ollama.prompt_eval_count,
        _fmt(ollama.prompt_eval_duration_ms),
        ollama.eval_count,
        _fmt(ollama.eval_duration_ms),
        _fmt(ollama.eval_tokens_per_sec),
        ollama.done_reason,
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"
