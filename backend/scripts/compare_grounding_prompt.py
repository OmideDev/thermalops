"""
Opt-in Phase 5C limited live comparison (≤3 scenarios).

Requires local PDF + FastEmbed + Ollama. Does not modify the production index.

  set RUN_RAG_GROUNDING_COMPARE=1
  python scripts/compare_grounding_prompt.py

Writes gitignored JSON under app/rag_eval/reports/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

REPORTS_DIR = BACKEND_ROOT / "app" / "rag_eval" / "reports"
SCENARIO_IDS = ("S05", "S13", "U02")


def main() -> int:
    if os.environ.get("RUN_RAG_GROUNDING_COMPARE") != "1":
        print(
            "Refusing: set RUN_RAG_GROUNDING_COMPARE=1 "
            "(local PDF + FastEmbed + Ollama).",
            file=sys.stderr,
        )
        return 2

    from app.rag.ask import ask_with_evidence
    from app.rag.embeddings import DEFAULT_MODEL_NAME, try_create_minilm_embedder
    from app.rag.ollama import DEFAULT_OLLAMA_MODEL, OllamaChatBackend
    from app.rag.search import build_index
    from app.rag.timing import AssistantRequestMetrics
    from app.rag_eval.dataset import load_dataset
    from app.services.assistant import (
        BACKEND_ROOT as SVC_ROOT,
        DEFAULT_ALLOWED_PAGES,
        load_eligible_chunks_from_manual,
    )

    baseline_path = REPORTS_DIR / "answer_eval_smoke_v1_results.json"
    baseline_by_id: dict[str, dict] = {}
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        for row in baseline.get("rows") or []:
            if row.get("eval_mode") == "production_index":
                baseline_by_id[row["scenario_id"]] = row

    dataset = load_dataset()
    by_id = {s["id"]: s for s in dataset["scenarios"]}
    scenarios = [by_id[sid] for sid in SCENARIO_IDS]

    pdf = SVC_ROOT / "data" / "manuals" / "460945ERT_iom.pdf"
    if not pdf.is_file():
        print(f"manual PDF missing: {pdf.name}", file=sys.stderr)
        return 3

    eligible, meta = load_eligible_chunks_from_manual(
        pdf, allowed_pages=DEFAULT_ALLOWED_PAGES
    )
    embedder = try_create_minilm_embedder(DEFAULT_MODEL_NAME)
    index = build_index(
        eligible,
        embedder,
        model_name=getattr(embedder, "model_name", DEFAULT_MODEL_NAME),
    )
    titles = {
        meta.get("document_id")
        or "460945ERT_iom.pdf": "Energy Recovery Ventilator IOM"
    }
    llm = OllamaChatBackend(model=DEFAULT_OLLAMA_MODEL)

    comparisons: list[dict] = []
    for scenario in scenarios:
        metrics = AssistantRequestMetrics()
        t0 = time.perf_counter()
        result = ask_with_evidence(
            scenario["question"],
            index,
            embedder,
            llm,
            top_k=3,
            document_titles=titles,
            metrics=metrics,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        prev = baseline_by_id.get(scenario["id"]) or {}
        ollama = getattr(llm, "last_timing", None)
        comparisons.append(
            {
                "scenario_id": scenario["id"],
                "question": scenario["question"],
                "expected_behavior": scenario["expected_behavior"],
                "previous_answer": prev.get("answer"),
                "previous_eval_count": (prev.get("ollama_timing") or {}).get(
                    "eval_count"
                ),
                "previous_latency_ms": prev.get("latency_ms"),
                "new_answer": result["answer"],
                "new_sources": result["sources"],
                "new_outcome": metrics.outcome,
                "new_latency_ms": latency_ms,
                "new_llm_wall_ms": metrics.llm_wall_ms,
                "new_eval_count": getattr(ollama, "eval_count", None),
                "new_prompt_eval_count": getattr(ollama, "prompt_eval_count", None),
                "new_ollama_total_ms": getattr(ollama, "total_duration_ms", None),
                "citation_pages": [
                    s.get("page") for s in result["sources"] if isinstance(s, dict)
                ],
            }
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "phase5c_grounding_compare_local.json"
    payload = {
        "model": DEFAULT_OLLAMA_MODEL,
        "scenario_ids": list(SCENARIO_IDS),
        "index_size": index.size,
        "note": (
            "Shorter is not automatically better. Judge claim support vs evidence. "
            "Prompt changes do not guarantee factual grounding."
        ),
        "comparisons": comparisons,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": out_path.name, "n": len(comparisons)}, indent=2))
    for item in comparisons:
        print(
            f"\n=== {item['scenario_id']} ===\n"
            f"PREV ({item.get('previous_eval_count')} tok): "
            f"{(item.get('previous_answer') or '')[:200]}\n"
            f"NEW  ({item.get('new_eval_count')} tok, "
            f"{item.get('new_latency_ms'):.0f} ms): "
            f"{(item.get('new_answer') or '')[:200]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
