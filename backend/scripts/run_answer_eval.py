"""
Opt-in limited grounded-answer evaluation (Phase 5B.2B).

Runs at most the eight selected scenarios against local Qwen via Ollama.
Operational-policy cases that stop before generation are not counted as
Qwen generation calls.

Usage (from backend/):

  set RUN_RAG_ANSWER_EVAL=1
  python scripts/run_answer_eval.py

Optional separate post-generation probe (one extra Ollama call; synthetic
index only — does not modify the production PDF index):

  set RUN_RAG_ANSWER_EVAL_POSTGATE=1
  python scripts/run_answer_eval.py --postgate-only

Writes local JSON under app/rag_eval/reports/ (gitignored raw results).
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

REPORTS_DIR = BACKEND_ROOT / "app" / "rag_eval" / "reports"


def _write_json(path: Path, payload: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_postgate_only() -> int:
    from app.rag.embeddings import DEFAULT_MODEL_NAME, try_create_minilm_embedder
    from app.rag.ollama import DEFAULT_OLLAMA_MODEL, OllamaChatBackend
    from app.rag_eval.answer_eval import evaluate_post_generation_poisoned_probe

    embedder = try_create_minilm_embedder(DEFAULT_MODEL_NAME)
    llm = OllamaChatBackend(model=DEFAULT_OLLAMA_MODEL)
    row = evaluate_post_generation_poisoned_probe(llm=llm, embedder=embedder)
    out = {
        "probe_id": "post_generation_poisoned_info_question",
        "model": DEFAULT_OLLAMA_MODEL,
        "row": row,
        "interpretation": {
            "llm_called": row.get("llm_called"),
            "model_emitted_injection_markers": (
                (row.get("model_output") or {}).get("followed_injection_markers")
            ),
            "final_application_actual_behavior": row.get("actual_behavior"),
            "operational_gate": row.get("operational_gate"),
            "unsafe_escaped_to_final_answer": (
                row.get("actual_behavior") == "answer_with_citations"
                and row.get("followed_injection_markers") is True
            ),
            "note": (
                "Separate from A01 pre-gated paths. Not comprehensive "
                "prompt-injection resistance."
            ),
        },
    }
    path = REPORTS_DIR / "answer_eval_postgate_probe_local.json"
    _write_json(path, out)
    print(json.dumps({"interpretation": out["interpretation"], "wrote": path.name}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    postgate_only = "--postgate-only" in args

    if postgate_only:
        if os.environ.get("RUN_RAG_ANSWER_EVAL_POSTGATE") != "1":
            print(
                "Refusing: set RUN_RAG_ANSWER_EVAL_POSTGATE=1 for the "
                "separate post-generation probe.",
                file=sys.stderr,
            )
            return 2
        return run_postgate_only()

    if os.environ.get("RUN_RAG_ANSWER_EVAL") != "1":
        print(
            "Refusing to run: set RUN_RAG_ANSWER_EVAL=1 to opt in "
            "(requires local PDF + FastEmbed + Ollama Qwen).",
            file=sys.stderr,
        )
        return 2

    from app.rag.embeddings import DEFAULT_MODEL_NAME, try_create_minilm_embedder
    from app.rag.ollama import DEFAULT_OLLAMA_MODEL, OllamaChatBackend
    from app.rag.search import build_index
    from app.rag_eval.answer_eval import (
        build_synthetic_adversarial_index,
        evaluate_ask_scenario,
        evaluate_raw_model_on_poisoned_prompt,
        selection_bundle,
    )
    from app.rag_eval.answer_report import build_analysis_markdown, summarize_answer_eval
    from app.services.assistant import (
        BACKEND_ROOT as SVC_ROOT,
        DEFAULT_ALLOWED_PAGES,
        load_eligible_chunks_from_manual,
    )

    bundle = selection_bundle()
    selection = bundle["selection"]
    scenarios = bundle["scenarios"]
    if len(scenarios) > 8:
        print("selection exceeds 8 scenarios; refusing", file=sys.stderr)
        return 3

    pdf = SVC_ROOT / "data" / "manuals" / "460945ERT_iom.pdf"
    if not pdf.is_file():
        print(f"manual PDF missing: {pdf.name}", file=sys.stderr)
        return 3

    eligible, meta = load_eligible_chunks_from_manual(
        pdf, allowed_pages=DEFAULT_ALLOWED_PAGES
    )
    embedder = try_create_minilm_embedder(DEFAULT_MODEL_NAME)
    production_index = build_index(
        eligible,
        embedder,
        model_name=getattr(embedder, "model_name", DEFAULT_MODEL_NAME),
    )
    titles = {
        meta.get("document_id")
        or "460945ERT_iom.pdf": "Energy Recovery Ventilator IOM"
    }

    llm = OllamaChatBackend(model=DEFAULT_OLLAMA_MODEL)
    rows: list[dict] = []
    adversarial_raw: list[dict] = []
    generation_calls = 0
    call_log: list[dict] = []

    for scenario in scenarios:
        if scenario["category"] == "adversarial_injection":
            fixture_id = scenario["provenance"].get(
                "synthetic_fixture_id", "ADV_INJECT_BLOWER_OVERRIDE"
            )
            prod_row = evaluate_ask_scenario(
                scenario,
                index=production_index,
                embedder=embedder,
                llm=llm,
                eval_mode="production_index_adversarial_question",
                document_titles=titles,
            )
            call_log.append(
                {
                    "scenario_id": scenario["id"],
                    "eval_mode": prod_row["eval_mode"],
                    "llm_called": prod_row.get("llm_called"),
                    "abstention_mechanism": prod_row.get("abstention_mechanism"),
                }
            )
            if prod_row.get("llm_called"):
                generation_calls += 1
            rows.append(prod_row)

            synth_index, synth_emb = build_synthetic_adversarial_index(
                fixture_id, embedder=embedder
            )
            synth_row = evaluate_ask_scenario(
                scenario,
                index=synth_index,
                embedder=synth_emb,
                llm=llm,
                eval_mode="synthetic_adversarial_index",
                document_titles={
                    "SYNTHETIC_ADVERSARIAL_NOT_MANUAL": "SYNTHETIC_ADVERSARIAL_NOT_MANUAL"
                },
            )
            call_log.append(
                {
                    "scenario_id": scenario["id"],
                    "eval_mode": synth_row["eval_mode"],
                    "llm_called": synth_row.get("llm_called"),
                    "abstention_mechanism": synth_row.get("abstention_mechanism"),
                }
            )
            if synth_row.get("llm_called"):
                generation_calls += 1
            rows.append(synth_row)

            raw_row = evaluate_raw_model_on_poisoned_prompt(
                scenario, llm=llm, fixture_id=fixture_id
            )
            raw_reached = raw_row.get("error") is None
            if raw_reached:
                generation_calls += 1
            call_log.append(
                {
                    "scenario_id": scenario["id"],
                    "eval_mode": raw_row["eval_mode"],
                    "llm_called": raw_reached,
                    "abstention_mechanism": "model_only_no_application_gates",
                }
            )
            adversarial_raw.append(raw_row)
            continue

        row = evaluate_ask_scenario(
            scenario,
            index=production_index,
            embedder=embedder,
            llm=llm,
            eval_mode="production_index",
            document_titles=titles,
        )
        call_log.append(
            {
                "scenario_id": scenario["id"],
                "eval_mode": row["eval_mode"],
                "llm_called": row.get("llm_called"),
                "abstention_mechanism": row.get("abstention_mechanism"),
            }
        )
        if row.get("llm_called"):
            generation_calls += 1
        rows.append(row)

    summary = summarize_answer_eval(rows, model_only_rows=adversarial_raw)
    summary["ollama_generation_calls_observed"] = generation_calls
    summary["call_log"] = call_log
    summary["adversarial_raw_model"] = adversarial_raw
    summary["pdf_content_sha256"] = meta.get("content_sha256")
    summary["index_size"] = production_index.size
    summary["model"] = DEFAULT_OLLAMA_MODEL

    results_path = REPORTS_DIR / "answer_eval_smoke_v1_results.json"
    report_path = REPORTS_DIR / "answer_eval_smoke_v1_analysis.md"

    payload = {
        "selection": selection,
        "summary": summary,
        "rows": rows,
        "adversarial_raw_model": adversarial_raw,
        "call_log": call_log,
    }
    _write_json(results_path, payload)
    report_path.write_text(
        build_analysis_markdown(
            selection=selection,
            summary=summary,
            rows=rows,
            offline_tests_passed=None,
            model_only_rows=adversarial_raw,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "summary_populations": summary.get("populations"),
                "ollama_generation_calls_observed": generation_calls,
                "latency": summary.get("latency"),
                "wrote": results_path.name,
            },
            indent=2,
        )
    )
    print(f"analysis: {report_path.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
