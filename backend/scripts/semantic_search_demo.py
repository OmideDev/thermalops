"""
Local semantic search demo over eligible Greenheck narrative chunks.

Indexes ONLY pages 15, 29, and 33. Excludes page 31 and any chunk that is not
retrieval-eligible. Uses a local MiniLM ONNX model (no paid API).

Usage (from the backend folder, after installing requirements-rag.txt):

    python scripts/semantic_search_demo.py
    python scripts/semantic_search_demo.py --query "How does frost control work?"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.chunk import PageInput
from app.rag.embeddings import DEFAULT_MODEL_NAME, try_create_minilm_embedder
from app.rag.search import (
    build_index,
    collect_eligible_chunks_from_pages,
    search_index,
)

DEFAULT_PDF = BACKEND_ROOT / "data" / "manuals" / "460945ERT_iom.pdf"
ALLOWED_PAGES = {15, 29, 33}
DEFAULT_QUERIES = [
    "How does the dirty filter sensor work?",
    "What should I check during filter maintenance?",
    "How does frost control work?",
    "What does the CO2 sensor do?",
    (
        "What is the corrective action in the page 31 troubleshooting table "
        "for a blower that fails to operate?"
    ),
]


def _load_extract_module():
    script = BACKEND_ROOT / "scripts" / "extract_pdf_text.py"
    spec = importlib.util.spec_from_file_location("extract_pdf_text", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_demo_index(pdf_path: Path):
    extract = _load_extract_module()
    extraction = extract.extract_pages(pdf_path, sorted(ALLOWED_PAGES))
    metadata = extraction["metadata"]
    pages = [
        PageInput(page_number=page["page_number"], text=page["normalized_text"])
        for page in extraction["pages"]
    ]
    eligible = collect_eligible_chunks_from_pages(
        pages,
        document_id=metadata["document_id"],
        content_sha256=metadata["content_sha256"],
        allowed_pages=ALLOWED_PAGES,
    )
    # Hard safety: never index page 31 even if eligibility slips.
    eligible = [chunk for chunk in eligible if chunk["page_number"] != 31]

    embedder = try_create_minilm_embedder(DEFAULT_MODEL_NAME)
    index = build_index(
        eligible,
        embedder,
        model_name=DEFAULT_MODEL_NAME,
    )
    return index, embedder, metadata, eligible


def print_results(query: str, results: list[dict]) -> None:
    print("=" * 72)
    print(f"Query: {query}")
    print(
        "Note: similarity scores are cosine rankings in roughly [-1, 1], "
        "not calibrated confidence probabilities."
    )
    if not results:
        print("No results (empty index).")
        return
    for item in results:
        print("-" * 72)
        print(f"Rank: {item['rank']}")
        print(f"Similarity: {item['similarity']:.4f}")
        print(f"document_id: {item['document_id']}")
        print(f"page_number: {item['page_number']}")
        print(f"section_heading: {item['section_heading']}")
        print(f"chunk_id: {item['chunk_id']}")
        print(f"preview: {item['preview']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local MiniLM semantic search over eligible HVAC chunks.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Path to the local Greenheck PDF (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Optional query (repeatable). Defaults to the evaluation set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return (default: 3)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pdf_path = args.pdf
    if not pdf_path.is_absolute():
        candidate = (Path.cwd() / pdf_path).resolve()
        pdf_path = candidate if candidate.exists() else (BACKEND_ROOT / pdf_path)

    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    print(f"Building index from {pdf_path} (pages {sorted(ALLOWED_PAGES)})...")
    try:
        index, embedder, metadata, eligible = build_demo_index(pdf_path)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Install optional deps: python -m pip install -r requirements-rag.txt",
            file=sys.stderr,
        )
        return 2

    print(f"document_id: {metadata['document_id']}")
    print(f"model: {DEFAULT_MODEL_NAME}")
    print(f"embedding_dimension: {index.dimension}")
    print(f"indexed_eligible_chunks: {index.size}")
    print(f"eligible_pages: {sorted({c['page_number'] for c in eligible})}")
    print()

    queries = args.query if args.query else DEFAULT_QUERIES
    for query in queries:
        results = search_index(index, query, embedder, top_k=args.top_k)
        print_results(query, results)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
