"""Offline unit tests for in-memory semantic search (no model download)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.chunk import PageInput
from app.rag.search import (
    DeterministicHashEmbedder,
    InMemorySemanticIndex,
    build_index,
    collect_eligible_chunks_from_pages,
    filter_eligible_chunks,
    search_index,
)


DOC_ID = "sample_manual.pdf"
DOC_SHA = "c" * 64
BACKEND_ROOT = Path(__file__).resolve().parents[1]
FAKE_MODEL = DeterministicHashEmbedder.model_name


def _eligible_chunk(**overrides: object) -> dict:
    base = {
        "chunk_id": "chunk-a",
        "document_id": DOC_ID,
        "content_sha256": DOC_SHA,
        "page_number": 15,
        "chunk_index": 0,
        "section_heading": "Frost Control",
        "text": "frost control wheel frosting timed exhaust narrative",
        "char_count": 52,
        "retrieval_eligibility": "eligible",
        "safe_for_retrieval": True,
    }
    base.update(overrides)
    return base


def _eligible_frost_and_filter_chunks() -> list[dict]:
    pages = [
        PageInput(
            page_number=15,
            text=(
                "Frost Control\n\n"
                "Frost control prevents moisture condensation and frosting "
                "on the energy recovery wheel using timed exhaust options.\n\n"
                "Dirty Filter Sensor\n\n"
                "Dirty filter sensors monitor pressure drop across the outdoor "
                "air filters and exhaust air filters."
            ),
        ),
        PageInput(
            page_number=29,
            text=(
                "Internal Filter\n\n"
                "Check filters during routine maintenance and replace dirty "
                "filters to keep airflow healthy."
            ),
        ),
        PageInput(
            page_number=31,
            text=(
                "Troubleshooting – Unit\n"
                "Symptom\n"
                "Possible Cause\n"
                "Corrective Action\n"
                "Blower fails to\n"
                "operate\n"
                "Blown fuse or open circuit breaker.\n"
                "Replace fuse or reset circuit breaker and check amps."
            ),
        ),
    ]
    return collect_eligible_chunks_from_pages(
        pages,
        document_id=DOC_ID,
        content_sha256=DOC_SHA,
        allowed_pages={15, 29, 33},
    )


def test_eligibility_filter_keeps_only_eligible_flags():
    chunks = [
        {
            "text": "ok",
            "retrieval_eligibility": "eligible",
            "safe_for_retrieval": True,
        },
        {
            "text": "bad",
            "retrieval_eligibility": "excluded",
            "safe_for_retrieval": False,
        },
        {
            "text": "also bad",
            "retrieval_eligibility": "eligible",
            "safe_for_retrieval": False,
        },
    ]
    kept = filter_eligible_chunks(chunks)
    assert len(kept) == 1
    assert kept[0]["text"] == "ok"


def test_excluded_table_content_cannot_enter_index():
    eligible = _eligible_frost_and_filter_chunks()
    assert all(chunk["page_number"] != 31 for chunk in eligible)
    joined = "\n".join(chunk["text"] for chunk in eligible)
    assert "Blown fuse or open circuit breaker." not in joined
    assert "Corrective Action" not in joined

    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index(eligible, embedder, model_name=FAKE_MODEL)
    assert index.size == len(eligible)
    for chunk in index.chunks:
        assert chunk.page_number in {15, 29}
        assert chunk.safe_for_retrieval is True
        assert chunk.retrieval_eligibility == "eligible"


def test_correct_ranking_with_controlled_vectors():
    chunks = [
        _eligible_chunk(
            chunk_id="a",
            section_heading="Frost Control",
            text="frost control wheel frosting timed exhaust",
            char_count=40,
        ),
        _eligible_chunk(
            chunk_id="b",
            chunk_index=1,
            section_heading="Dirty Filter Sensor",
            text="dirty filter sensor pressure drop across filters",
            char_count=48,
        ),
        _eligible_chunk(
            chunk_id="c",
            page_number=29,
            chunk_index=0,
            section_heading="Internal Filter",
            text="replace dirty filters during maintenance inspection",
            char_count=50,
        ),
    ]
    embedder = DeterministicHashEmbedder(dimension=64)
    index = build_index(chunks, embedder, model_name=FAKE_MODEL)
    results = search_index(
        index,
        "dirty filter sensor pressure",
        embedder,
        top_k=2,
    )
    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[0]["similarity"] >= results[1]["similarity"]
    assert results[0]["section_heading"] == "Dirty Filter Sensor"
    assert results[0]["chunk_id"] == "b"


def test_stable_mapping_from_vectors_back_to_metadata():
    chunks = _eligible_frost_and_filter_chunks()
    embedder = DeterministicHashEmbedder()
    index = build_index(chunks, embedder, model_name=FAKE_MODEL)
    assert index.size >= 1
    for stored, source in zip(index.chunks, filter_eligible_chunks(chunks), strict=True):
        assert stored.chunk_id == source["chunk_id"]
        assert stored.page_number == source["page_number"]
        assert stored.section_heading == source["section_heading"]
        assert stored.text == source["text"]


def test_empty_corpus_returns_no_hits():
    embedder = DeterministicHashEmbedder()
    index = build_index([], embedder, model_name=FAKE_MODEL)
    assert index.size == 0
    assert search_index(index, "frost control", embedder, top_k=3) == []


def test_invalid_or_empty_query_raises():
    embedder = DeterministicHashEmbedder()
    index = build_index([_eligible_chunk()], embedder, model_name=FAKE_MODEL)
    with pytest.raises(ValueError, match="non-empty"):
        search_index(index, "   ", embedder, top_k=3)
    with pytest.raises(ValueError, match="non-empty"):
        search_index(index, "", embedder, top_k=3)


def test_top_k_behavior():
    chunks = [
        _eligible_chunk(
            chunk_id=f"id-{index}",
            chunk_index=index,
            text=f"frost control detail number {index} wheel sensor",
            char_count=40,
        )
        for index in range(5)
    ]
    embedder = DeterministicHashEmbedder(dimension=48)
    index = build_index(chunks, embedder, model_name=FAKE_MODEL)
    top2 = search_index(index, "frost control wheel", embedder, top_k=2)
    top5 = search_index(index, "frost control wheel", embedder, top_k=5)
    assert len(top2) == 2
    assert len(top5) == 5
    assert [item["rank"] for item in top2] == [1, 2]
    with pytest.raises(ValueError):
        search_index(index, "frost", embedder, top_k=0)


def test_allowed_pages_exclude_page_31_even_if_present():
    eligible = collect_eligible_chunks_from_pages(
        [
            PageInput(
                page_number=31,
                text=(
                    "Troubleshooting – Unit\n\n"
                    "This narrative should never be indexed because page 31 "
                    "is outside the allowed embeddings experiment pages."
                ),
            ),
            PageInput(
                page_number=15,
                text=(
                    "Frost Control\n\n"
                    "Frost control prevents moisture condensation on the wheel."
                ),
            ),
        ],
        document_id=DOC_ID,
        content_sha256=DOC_SHA,
        allowed_pages={15, 29, 33},
    )
    assert eligible
    assert all(chunk["page_number"] != 31 for chunk in eligible)
    assert all(chunk["page_number"] == 15 for chunk in eligible)


def test_fake_embedder_reproducible_across_pythonhashseed():
    """SHA-256 fake embedder must ignore PYTHONHASHSEED randomization."""
    script = (
        "import json,sys;"
        "sys.path.insert(0, r'%s');"
        "from app.rag.search import DeterministicHashEmbedder;"
        "e=DeterministicHashEmbedder(32);"
        "print(json.dumps(e.embed(['frost control dirty filter sensor'])))"
    ) % str(BACKEND_ROOT).replace("\\", "\\\\")

    def run_with_seed(seed: str) -> list[list[float]]:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return json.loads(completed.stdout.strip())

    first = run_with_seed("0")
    second = run_with_seed("12345")
    assert first == second
    # Also matches an in-process embedder in this pytest process.
    local = DeterministicHashEmbedder(32).embed(
        ["frost control dirty filter sensor"]
    )
    assert first == local


def test_vector_dimension_mismatch_fails_early():
    chunk = _eligible_chunk()
    with pytest.raises(ValueError, match="expected dimension"):
        InMemorySemanticIndex(
            [chunk],
            [[0.1, 0.2]],
            model_name=FAKE_MODEL,
            dimension=4,
        )


def test_vector_count_mismatch_fails_early():
    chunk = _eligible_chunk()
    with pytest.raises(ValueError, match="expected 1 vector"):
        InMemorySemanticIndex(
            [chunk],
            [[0.1, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0]],
            model_name=FAKE_MODEL,
            dimension=4,
        )


def test_non_finite_vector_component_fails_early():
    chunk = _eligible_chunk()
    with pytest.raises(ValueError, match="not finite"):
        InMemorySemanticIndex(
            [chunk],
            [[0.1, float("nan"), 0.0, 0.0]],
            model_name=FAKE_MODEL,
            dimension=4,
        )


def test_zero_document_vector_rejected():
    chunk = _eligible_chunk()
    with pytest.raises(ValueError, match="all-zero"):
        InMemorySemanticIndex(
            [chunk],
            [[0.0, 0.0, 0.0, 0.0]],
            model_name=FAKE_MODEL,
            dimension=4,
        )


def test_zero_query_vector_returns_no_neighbors():
    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible_chunk()], embedder, model_name=FAKE_MODEL)
    # Punctuation-only query tokenizes to nothing → all-zero fake embedding.
    assert search_index(index, "... !!!", embedder, top_k=3) == []
    assert index.search([0.0] * index.dimension, top_k=3) == []


def test_query_vector_wrong_dimension_fails():
    embedder = DeterministicHashEmbedder(dimension=32)
    index = build_index([_eligible_chunk()], embedder, model_name=FAKE_MODEL)
    with pytest.raises(ValueError, match="query_vector"):
        index.search([0.1, 0.2], top_k=3)


def test_constructor_rejects_ineligible_chunks():
    bad = _eligible_chunk(
        retrieval_eligibility="excluded",
        safe_for_retrieval=False,
        text="should not enter index",
    )
    with pytest.raises(ValueError, match="not retrieval-eligible"):
        InMemorySemanticIndex(
            [bad],
            [[0.1, 0.2, 0.3, 0.4]],
            model_name=FAKE_MODEL,
            dimension=4,
        )


def test_model_name_mismatch_on_build_and_query():
    embedder = DeterministicHashEmbedder(dimension=32)
    with pytest.raises(ValueError, match="does not match"):
        build_index([_eligible_chunk()], embedder, model_name="other-model")

    index = build_index([_eligible_chunk()], embedder, model_name=FAKE_MODEL)

    class OtherModel:
        model_name = "totally-different"
        dimension = 32

        def embed(self, texts):
            return embedder.embed(texts)

    with pytest.raises(ValueError, match="does not match"):
        search_index(index, "frost control", OtherModel(), top_k=3)
