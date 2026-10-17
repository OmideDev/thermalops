"""
LLM backends for grounded assistant answers.

Retrieval stays independent of any specific model vendor. Phase 4A ships a
configurable FakeLLMBackend for offline tests. An Ollama (or other) backend
can be added later behind the same Protocol without rewriting ask_with_evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol


class LLMBackend(Protocol):
    """
    Minimal chat completion interface used by grounded answer generation.

    Implementations must return a single assistant message string. Structured
    JSON parsing and citation validation happen in the ask service, not here.
    """

    def complete(self, *, system: str, user: str) -> str:
        """Return the model reply for a system + user message pair."""


class FakeLLMBackend:
    """
    Configurable test-only LLM.

    Responses may be:
    - a fixed sequence of reply strings (consumed in order), or
    - a callable ``handler(system, user) -> str`` for dynamic scripts.

    Set ``error`` to raise on the next ``complete`` call (LLM failure path).
    Records the most recent prompts for injection / prompt-structure asserts.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        handler: Callable[[str, str], str] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if responses is not None and handler is not None:
            raise ValueError("provide responses or handler, not both")
        self._responses = list(responses or [])
        self._handler = handler
        self.error = error
        self.calls: list[dict[str, str]] = []
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, *, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        self.calls.append({"system": system, "user": user})
        if self.error is not None:
            raise self.error
        if self._handler is not None:
            return self._handler(system, user)
        if not self._responses:
            raise RuntimeError("FakeLLMBackend has no remaining scripted responses")
        return self._responses.pop(0)

    def reset_calls(self) -> None:
        self.calls.clear()
        self.last_system = None
        self.last_user = None
