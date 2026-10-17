"""RAG helpers for ThermalOps (normalize, chunk, local semantic search, ask)."""

from app.rag.ask import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    AskResponse,
    AssistantSource,
    ask_with_evidence,
)
from app.rag.chunk import PageInput, chunk_page, chunk_pages
from app.rag.llm import FakeLLMBackend, LLMBackend
from app.rag.normalize import normalize_text
from app.rag.search import (
    DeterministicHashEmbedder,
    InMemorySemanticIndex,
    build_index,
    collect_eligible_chunks_from_pages,
    filter_eligible_chunks,
    search_index,
)

__all__ = [
    "AskResponse",
    "AssistantSource",
    "DeterministicHashEmbedder",
    "FakeLLMBackend",
    "INSUFFICIENT_EVIDENCE_ANSWER",
    "InMemorySemanticIndex",
    "LLMBackend",
    "PageInput",
    "ask_with_evidence",
    "build_index",
    "chunk_page",
    "chunk_pages",
    "collect_eligible_chunks_from_pages",
    "filter_eligible_chunks",
    "normalize_text",
    "search_index",
]
