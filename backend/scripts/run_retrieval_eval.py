"""
Opt-in local retrieval evaluation (no Ollama).

Requires the local Greenheck PDF and FastEmbed MiniLM weights.

Usage (from backend/):

  set RUN_RETRIEVAL_EVAL=1
  python scripts/run_retrieval_eval.py

Default offline pytest does not execute this script.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def main() -> int:
    if os.environ.get("RUN_RETRIEVAL_EVAL") != "1":
        print(
            "Refusing to run: set RUN_RETRIEVAL_EVAL=1 to opt in "
            "(requires local PDF + FastEmbed).",
            file=sys.stderr,
        )
        return 2

    from app.rag.embeddings import DEFAULT_MODEL_NAME, try_create_minilm_embedder
    from app.rag.search import build_index
    from app.rag_eval.dataset import load_dataset
    from app.rag_eval.retrieval import evaluate_retrieval
    from app.services.assistant import (
        BACKEND_ROOT as SVC_ROOT,
        DEFAULT_ALLOWED_PAGES,
        load_eligible_chunks_from_manual,
    )

    dataset = load_dataset()
    pdf = SVC_ROOT / "data" / "manuals" / "460945ERT_iom.pdf"
    if not pdf.is_file():
        print(f"manual PDF missing: {pdf.name}", file=sys.stderr)
        return 3

    eligible, meta = load_eligible_chunks_from_manual(
        pdf, allowed_pages=DEFAULT_ALLOWED_PAGES
    )
    sha = meta.get("content_sha256")
    if sha and sha != dataset["content_sha256"]:
        print(
            "WARNING: PDF content_sha256 differs from dataset provenance "
            f"(pdf={sha}, dataset={dataset['content_sha256']})",
            file=sys.stderr,
        )

    embedder = try_create_minilm_embedder(DEFAULT_MODEL_NAME)
    index = build_index(
        eligible,
        embedder,
        model_name=getattr(embedder, "model_name", DEFAULT_MODEL_NAME),
    )
    report = evaluate_retrieval(
        dataset["scenarios"],
        index,
        embedder,
        top_k=3,
    )
    report["document_id"] = dataset["document_id"]
    report["dataset_id"] = dataset["dataset_id"]
    report["index_size"] = index.size
    report["embedding_model"] = dataset["embedding_model"]

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
