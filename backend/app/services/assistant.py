"""
Experimental grounded-assistant service (process-scoped).

Builds the semantic index once at startup when enabled. Uses blocking Ollama
HTTP under a per-process concurrency lock (one generation at a time). This
limit is not global across multiple worker processes.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.rag.ask import AskResponse, ask_with_evidence
from app.rag.chunk import PageInput
from app.rag.embeddings import DEFAULT_MODEL_NAME, try_create_minilm_embedder
from app.rag.llm import LLMBackend
from app.rag.ollama import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OllamaChatBackend,
)
from app.rag.search import (
    EmbeddingBackend,
    InMemorySemanticIndex,
    build_index,
    collect_eligible_chunks_from_pages,
)

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_PAGES: frozenset[int] = frozenset({15, 29, 33})
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class AssistantUnavailableError(RuntimeError):
    """Assistant disabled, missing manual/index, or not initialized."""


class AssistantBusyError(RuntimeError):
    """Another generation is already in progress in this process."""


class AssistantAskFn(Protocol):
    def __call__(
        self,
        question: str,
        index: InMemorySemanticIndex,
        embedder: EmbeddingBackend,
        llm: LLMBackend,
        *,
        top_k: int = 3,
        document_titles: dict[str, str] | None = None,
    ) -> AskResponse: ...


def _load_extract_module():
    script = BACKEND_ROOT / "scripts" / "extract_pdf_text.py"
    spec = importlib.util.spec_from_file_location("extract_pdf_text", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extract module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_eligible_chunks_from_manual(
    pdf_path: Path,
    *,
    allowed_pages: set[int] | frozenset[int] = DEFAULT_ALLOWED_PAGES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Extract, normalize, chunk, and filter eligible narrative chunks.

    Indexes only ``allowed_pages`` (default 15/29/33). Page 31 is always
    excluded even if somehow present.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"manual PDF not found: {path.name}")

    pages_set = set(allowed_pages)
    extract = _load_extract_module()
    extraction = extract.extract_pages(path, sorted(pages_set))
    metadata = extraction["metadata"]
    pages = [
        PageInput(page_number=page["page_number"], text=page["normalized_text"])
        for page in extraction["pages"]
    ]
    eligible = collect_eligible_chunks_from_pages(
        pages,
        document_id=metadata["document_id"],
        content_sha256=metadata["content_sha256"],
        allowed_pages=pages_set,
    )
    eligible = [chunk for chunk in eligible if chunk["page_number"] != 31]
    eligible = [chunk for chunk in eligible if chunk["page_number"] in pages_set]
    if not eligible:
        raise RuntimeError("no eligible chunks for assistant index")
    return eligible, metadata


@dataclass
class AssistantService:
    """
    Process-scoped grounded assistant.

    ``index_build_count`` increments only when an index is constructed (for
    tests verifying once-per-process behavior).
    """

    index: InMemorySemanticIndex
    embedder: EmbeddingBackend
    llm: LLMBackend
    document_titles: dict[str, str] = field(default_factory=dict)
    top_k: int = 3
    ask_fn: AssistantAskFn = field(default=ask_with_evidence)
    index_build_count: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _owns_llm_close: bool = False

    @classmethod
    def build_from_manual(
        cls,
        pdf_path: Path,
        *,
        allowed_pages: set[int] | frozenset[int] = DEFAULT_ALLOWED_PAGES,
        document_title: str | None = None,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        connect_timeout: float = 5.0,
        read_timeout: float = 180.0,
        top_k: int = 3,
        llm: LLMBackend | None = None,
        embedder: EmbeddingBackend | None = None,
    ) -> AssistantService:
        """Build MiniLM embeddings + index once; optionally inject fakes."""
        eligible, metadata = load_eligible_chunks_from_manual(
            pdf_path, allowed_pages=allowed_pages
        )
        document_id = str(metadata["document_id"])
        titles = {
            document_id: document_title or Path(document_id).stem,
        }

        if embedder is None:
            embedder = try_create_minilm_embedder(DEFAULT_MODEL_NAME)

        index = build_index(
            eligible,
            embedder,
            model_name=getattr(embedder, "model_name", DEFAULT_MODEL_NAME),
        )

        owns_llm = llm is None
        owned_llm: LLMBackend | None = None
        try:
            if llm is None:
                owned_llm = OllamaChatBackend(
                    base_url=ollama_base_url,
                    model=ollama_model,
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                )
                llm = owned_llm

            service = cls(
                index=index,
                embedder=embedder,
                llm=llm,
                document_titles=titles,
                top_k=top_k,
                index_build_count=1,
                _owns_llm_close=owns_llm,
            )
        except Exception:
            if owned_llm is not None and hasattr(owned_llm, "close"):
                try:
                    owned_llm.close()  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    logger.exception("error closing LLM after init failure")
            raise

        logger.info(
            "Assistant index ready: document_id=%s chunks=%s pages=%s",
            document_id,
            index.size,
            sorted({c["page_number"] for c in eligible}),
        )
        return service

    def ask(self, question: str) -> AskResponse:
        """
        Run grounded ask under a non-blocking per-process lock.

        Raises AssistantBusyError if another request holds the lock.
        Propagates TimeoutError / ConnectionError from the LLM backend.
        """
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            raise AssistantBusyError(
                "assistant is busy; only one generation is allowed per process"
            )
        try:
            return self.ask_fn(
                question,
                self.index,
                self.embedder,
                self.llm,
                top_k=self.top_k,
                document_titles=self.document_titles,
            )
        finally:
            self._lock.release()

    def close(self) -> None:
        if self._owns_llm_close and hasattr(self.llm, "close"):
            try:
                self.llm.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                logger.exception("error closing assistant LLM backend")


def try_create_assistant_from_settings(settings: Any) -> AssistantService | None:
    """
    Best-effort startup helper. Returns None when disabled or init fails.

    Never raises into the FastAPI lifespan caller for missing PDFs / models.
    """
    if not getattr(settings, "assistant_enabled", False):
        logger.info("Experimental assistant disabled (assistant_enabled=false)")
        return None

    raw_pages = getattr(settings, "assistant_allowed_pages", "15,29,33")
    try:
        allowed = {
            int(part.strip())
            for part in str(raw_pages).split(",")
            if part.strip()
        }
    except ValueError:
        logger.warning("invalid assistant_allowed_pages=%r; assistant disabled", raw_pages)
        return None
    if not allowed:
        logger.warning("empty assistant_allowed_pages; assistant disabled")
        return None

    manual = getattr(settings, "assistant_manual_path", None)
    if not manual:
        default = BACKEND_ROOT / "data" / "manuals" / "460945ERT_iom.pdf"
        manual_path = default
    else:
        manual_path = Path(str(manual))
        if not manual_path.is_absolute():
            manual_path = (BACKEND_ROOT / manual_path).resolve()

    try:
        return AssistantService.build_from_manual(
            manual_path,
            allowed_pages=allowed,
            document_title=getattr(settings, "assistant_document_title", None),
            ollama_base_url=getattr(
                settings, "assistant_ollama_base_url", DEFAULT_OLLAMA_BASE_URL
            ),
            ollama_model=getattr(
                settings, "assistant_ollama_model", DEFAULT_OLLAMA_MODEL
            ),
            connect_timeout=float(
                getattr(settings, "assistant_ollama_connect_timeout", 5.0)
            ),
            read_timeout=float(
                getattr(settings, "assistant_ollama_read_timeout", 180.0)
            ),
            top_k=int(getattr(settings, "assistant_top_k", 3)),
        )
    except Exception as exc:  # noqa: BLE001 - startup must not crash the API
        logger.warning(
            "Experimental assistant unavailable (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return None
