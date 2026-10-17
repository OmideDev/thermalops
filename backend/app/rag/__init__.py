"""RAG helpers for ThermalOps (normalize, chunk, local semantic search)."""

from app.rag.chunk import PageInput, chunk_page, chunk_pages
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
    "DeterministicHashEmbedder",
    "InMemorySemanticIndex",
    "PageInput",
    "build_index",
    "chunk_page",
    "chunk_pages",
    "collect_eligible_chunks_from_pages",
    "filter_eligible_chunks",
    "normalize_text",
    "search_index",
]
