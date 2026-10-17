"""
In-memory semantic search over eligible RAG chunks.

Educational prototype: cosine similarity over dense vectors. No vector DB,
no LLM synthesis. Embedding backends are injectable so unit tests can use a
deterministic fake without downloading models.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.rag.chunk import PageInput, chunk_pages


class EmbeddingBackend(Protocol):
    """Minimal embedding interface used by the search index."""

    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one dense vector per input text."""


@dataclass(frozen=True)
class IndexedChunk:
    """Chunk metadata stored beside its embedding vector."""

    chunk_id: str
    document_id: str
    content_sha256: str
    page_number: int
    chunk_index: int
    section_heading: str | None
    text: str
    char_count: int
    retrieval_eligibility: str
    safe_for_retrieval: bool


def is_eligible_chunk(chunk: dict) -> bool:
    """Return True only for retrieval-ready narrative chunks."""
    return (
        chunk.get("retrieval_eligibility") == "eligible"
        and chunk.get("safe_for_retrieval") is True
        and isinstance(chunk.get("text"), str)
        and bool(chunk["text"].strip())
    )


def filter_eligible_chunks(chunks: Sequence[dict]) -> list[dict]:
    """Keep only chunks that passed the Phase 2 quality gate."""
    return [chunk for chunk in chunks if is_eligible_chunk(chunk)]


def collect_eligible_chunks_from_pages(
    pages: Sequence[PageInput],
    *,
    document_id: str,
    content_sha256: str,
    allowed_pages: set[int] | None = None,
) -> list[dict]:
    """
    Chunk pages and return only eligible narrative chunks.

    ``allowed_pages`` limits which 1-based page numbers may enter the index
    (e.g. {15, 29, 33} for the Greenheck experiment).
    """
    selected = [
        page
        for page in pages
        if allowed_pages is None or page.page_number in allowed_pages
    ]
    result = chunk_pages(
        list(selected),
        document_id=document_id,
        content_sha256=content_sha256,
    )
    eligible = filter_eligible_chunks(result["chunks"])
    if allowed_pages is not None:
        eligible = [
            chunk for chunk in eligible if chunk["page_number"] in allowed_pages
        ]
    return eligible


def _is_zero_vector(vector: Sequence[float]) -> bool:
    return all(component == 0.0 for component in vector)


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _validate_vector(
    vector: Sequence[float],
    *,
    dimension: int,
    label: str,
    allow_zero: bool = False,
) -> list[float]:
    """Validate shape and finiteness; optionally reject the all-zero vector."""
    if len(vector) != dimension:
        raise ValueError(
            f"{label} has length {len(vector)}, expected dimension {dimension}"
        )
    values: list[float] = []
    for index, raw in enumerate(vector):
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} component {index} is not a number: {raw!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"{label} component {index} is not finite: {value!r}"
            )
        values.append(value)
    if not allow_zero and _is_zero_vector(values):
        raise ValueError(f"{label} is an all-zero vector (not meaningful)")
    return values


def _validate_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    dimension: int,
) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise ValueError(
            f"expected {expected_count} vector(s), got {len(vectors)}"
        )
    validated: list[list[float]] = []
    for index, vector in enumerate(vectors):
        validated.append(
            _validate_vector(
                vector,
                dimension=dimension,
                label=f"vector[{index}]",
                allow_zero=False,
            )
        )
    return validated


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity for equal-length vectors (assumes same space)."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _stable_token_bytes(token: str) -> bytes:
    """Process-stable digest (immune to PYTHONHASHSEED randomization)."""
    return hashlib.sha256(token.encode("utf-8")).digest()


class DeterministicHashEmbedder:
    """
    Fake embedding backend for offline unit tests.

    Maps each text to a fixed-size bag-of-token hash vector using SHA-256 so
    results are reproducible across processes and PYTHONHASHSEED values.
    Not a neural model — test-only.
    """

    model_name = "deterministic-hash-v1"

    def __init__(self, dimension: int = 32) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_l2_normalize(self._vectorize(text)) for text in texts]

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = _stable_token_bytes(token)
            bucket = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[bucket] += sign
        return vector


class InMemorySemanticIndex:
    """Stores eligible chunks and their embeddings for Top-K cosine search."""

    def __init__(
        self,
        chunks: Sequence[dict],
        vectors: Sequence[Sequence[float]],
        *,
        model_name: str,
        dimension: int,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if not model_name or not str(model_name).strip():
            raise ValueError("model_name must be a non-empty string")

        # Enforce the retrieval quality gate even if callers bypass build_index.
        ineligible = [
            chunk for chunk in chunks if not is_eligible_chunk(chunk)
        ]
        if ineligible:
            raise ValueError(
                f"{len(ineligible)} chunk(s) are not retrieval-eligible; "
                "only eligible chunks may enter InMemorySemanticIndex"
            )

        validated = _validate_vectors(
            vectors,
            expected_count=len(chunks),
            dimension=dimension,
        )

        self.model_name = str(model_name).strip()
        self.dimension = dimension
        self.chunks: list[IndexedChunk] = [
            IndexedChunk(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                content_sha256=chunk["content_sha256"],
                page_number=chunk["page_number"],
                chunk_index=chunk["chunk_index"],
                section_heading=chunk.get("section_heading"),
                text=chunk["text"],
                char_count=chunk["char_count"],
                retrieval_eligibility=chunk.get(
                    "retrieval_eligibility", "eligible"
                ),
                safe_for_retrieval=bool(chunk.get("safe_for_retrieval")),
            )
            for chunk in chunks
        ]
        self.vectors: list[list[float]] = [
            _l2_normalize(vector) for vector in validated
        ]

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: Sequence[float], *, top_k: int = 3) -> list[dict]:
        """
        Return Top-K matches ranked by cosine similarity.

        Scores are ranking similarities in roughly [-1, 1], not calibrated
        confidence probabilities.

        An all-zero / empty-meaning query vector returns no neighbors (avoids
        arbitrary ties at similarity 0 without using a score threshold).
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not self.chunks:
            return []

        query_values = _validate_vector(
            query_vector,
            dimension=self.dimension,
            label="query_vector",
            allow_zero=True,
        )
        if _is_zero_vector(query_values):
            return []

        query = _l2_normalize(query_values)
        scored: list[tuple[float, int]] = []
        for index, vector in enumerate(self.vectors):
            score = cosine_similarity(query, vector)
            scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict] = []
        for rank, (score, index) in enumerate(scored[:top_k], start=1):
            chunk = self.chunks[index]
            preview = chunk.text.replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:220] + "..."
            results.append(
                {
                    "rank": rank,
                    "similarity": score,
                    "document_id": chunk.document_id,
                    "page_number": chunk.page_number,
                    "section_heading": chunk.section_heading,
                    "chunk_id": chunk.chunk_id,
                    "char_count": chunk.char_count,
                    "preview": preview,
                    "text": chunk.text,
                }
            )
        return results


def build_index(
    eligible_chunks: Sequence[dict],
    embedder: EmbeddingBackend,
    *,
    model_name: str,
) -> InMemorySemanticIndex:
    """Embed eligible chunks and build an in-memory cosine index."""
    clean = filter_eligible_chunks(eligible_chunks)
    embedder_model = getattr(embedder, "model_name", None)
    if embedder_model is not None and str(embedder_model) != str(model_name):
        raise ValueError(
            f"embedder model_name {embedder_model!r} does not match "
            f"requested model_name {model_name!r}"
        )

    if not clean:
        return InMemorySemanticIndex(
            [],
            [],
            model_name=model_name,
            dimension=embedder.dimension,
        )

    vectors = embedder.embed([chunk["text"] for chunk in clean])
    # Full validation (count, dimension, finiteness, non-zero) happens in the
    # index constructor; fail early here too for clearer embedder errors.
    _validate_vectors(
        vectors,
        expected_count=len(clean),
        dimension=embedder.dimension,
    )
    return InMemorySemanticIndex(
        clean,
        vectors,
        model_name=model_name,
        dimension=embedder.dimension,
    )


def search_index(
    index: InMemorySemanticIndex,
    query: str,
    embedder: EmbeddingBackend,
    *,
    top_k: int = 3,
) -> list[dict]:
    """Embed a query and run Top-K cosine search."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    embedder_model = getattr(embedder, "model_name", None)
    if embedder_model is not None and str(embedder_model) != index.model_name:
        raise ValueError(
            f"query embedder model_name {embedder_model!r} does not match "
            f"index model_name {index.model_name!r}"
        )

    query_vector = embedder.embed([query.strip()])[0]
    return index.search(query_vector, top_k=top_k)
