"""Aggregate answer-evaluation rows into human-review-friendly summaries."""

from __future__ import annotations

from statistics import median
from typing import Any, Sequence

from app.rag_eval.answer_scoring import (
    ACTUAL_ABSTAIN_INSUFFICIENT,
    ACTUAL_ABSTAIN_OPERATIONAL,
    ACTUAL_ANSWER,
    fraction,
)


def _latency_stats(values: Sequence[float]) -> dict[str, Any]:
    nums = [float(v) for v in values]
    if not nums:
        return {"n": 0, "mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
    return {
        "n": len(nums),
        "mean_ms": sum(nums) / float(len(nums)),
        "median_ms": float(median(nums)),
        "min_ms": min(nums),
        "max_ms": max(nums),
    }


def _ollama_inference_ms(row: dict[str, Any]) -> float | None:
    """Prefer Ollama total_duration_ms; fall back to llm_wall_ms when present."""
    timing = row.get("ollama_timing") or {}
    if isinstance(timing, dict):
        total = timing.get("total_duration_ms")
        if isinstance(total, (int, float)):
            return float(total)
    wall = row.get("llm_wall_ms")
    if isinstance(wall, (int, float)):
        return float(wall)
    return None


def summarize_answer_eval(
    rows: Sequence[dict[str, Any]],
    *,
    model_only_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Separate automatic structural outcomes from human factual review.

    Percentages always include numerator/denominator. Zero denominators yield
    rate=None (never silently 0% or 100%).

    Latency is reported in separate populations so pre-gated (~0 ms) rows are
    never mixed into LLM inference latency.
    """
    application_rows = [
        r for r in rows if r.get("eval_mode") != "raw_model_poisoned_context"
    ]
    supported = [
        r for r in application_rows if r.get("expected_behavior") == "answer_from_evidence"
    ]
    unsupported = [
        r
        for r in application_rows
        if r.get("expected_behavior") == "abstain_insufficient_evidence"
    ]
    policy = [
        r
        for r in application_rows
        if r.get("expected_behavior") == "abstain_operational_safety"
    ]
    adversarial_app = [
        r
        for r in application_rows
        if r.get("expected_behavior") == "adversarial_measure_separately"
    ]

    supported_structural = [r for r in supported if r.get("application_pass") is True]
    supported_needs_review = [r for r in supported if r.get("needs_human_review")]
    unsupported_ok = [r for r in unsupported if r.get("application_pass") is True]
    policy_ok = [r for r in policy if r.get("application_pass") is True]

    citation_ok = [
        r
        for r in supported
        if r.get("citation_page_overlap_expected") is True
        and r.get("actual_behavior") == ACTUAL_ANSWER
    ]
    citation_denom = [
        r for r in supported if r.get("actual_behavior") == ACTUAL_ANSWER
    ]

    llm_app_calls = [r for r in application_rows if r.get("llm_called") is True]
    pre_gate = [
        r
        for r in application_rows
        if r.get("operational_gate") in {"pre_generation", "pre_generation_inferred"}
    ]
    post_gate = [
        r
        for r in application_rows
        if r.get("operational_gate") in {"post_generation", "post_generation_inferred"}
    ]
    escaped_unsafe = [
        r
        for r in application_rows
        if r.get("followed_injection_markers")
        and r.get("actual_behavior") == ACTUAL_ANSWER
    ]

    # Full request latency: all application rows (includes pre-gate ~0 ms).
    full_request_latencies = [
        float(r["latency_ms"])
        for r in application_rows
        if isinstance(r.get("latency_ms"), (int, float))
    ]
    # Actual inference / generation latency: only rows that called the LLM.
    inference_latencies = [
        ms
        for r in llm_app_calls
        for ms in [_ollama_inference_ms(r)]
        if ms is not None
    ]
    generation_request_latencies = [
        float(r["latency_ms"])
        for r in llm_app_calls
        if isinstance(r.get("latency_ms"), (int, float))
    ]
    pre_gate_latencies = [
        float(r["latency_ms"])
        for r in pre_gate
        if isinstance(r.get("latency_ms"), (int, float))
    ]

    model_only = list(model_only_rows or [])
    model_only_injection = [
        r for r in model_only if r.get("followed_injection_markers") is True
    ]

    failures = [
        {
            "scenario_id": r.get("scenario_id"),
            "category": r.get("category"),
            "eval_mode": r.get("eval_mode"),
            "expected_behavior": r.get("expected_behavior"),
            "actual_behavior": r.get("actual_behavior"),
            "application_pass": r.get("application_pass"),
            "notes": r.get("notes"),
            "error": r.get("error"),
        }
        for r in application_rows
        if r.get("application_pass") is False or r.get("error")
    ]

    return {
        "n_application_rows": len(application_rows),
        "n_model_only_rows": len(model_only),
        "populations": {
            "selected_scenarios": 8,
            "application_rows": len(application_rows),
            "application_rows_note": (
                "A01 is reported twice at application level "
                "(production_index_adversarial_question + synthetic_adversarial_index), "
                "yielding 9 application rows from 8 selected scenario IDs."
            ),
            "model_only_rows": len(model_only),
            "llm_called_application_rows": len(llm_app_calls),
            "pre_generation_blocked_rows": len(pre_gate),
            "post_generation_blocked_rows": len(post_gate),
        },
        "structural_validity_supported": fraction(
            len(supported_structural), len(supported)
        ),
        "citation_page_overlap_among_answers": fraction(
            len(citation_ok), len(citation_denom)
        ),
        "claim_level_factual_support": {
            "status": "needs_human_review",
            "supported_needing_review": fraction(
                len(supported_needs_review), len(supported)
            ),
            "note": (
                "Automatic checks do not score claim-level factual support. "
                "Structural answer+citations ≠ verified factual correctness."
            ),
        },
        "unsupported_abstention_application": fraction(
            len(unsupported_ok), len(unsupported)
        ),
        "operational_policy_pre_generation": fraction(len(policy_ok), len(policy)),
        "llm_generation_among_application_rows": fraction(
            len(llm_app_calls), len(application_rows)
        ),
        "pre_generation_operational_gates": {
            "numerator": len(pre_gate),
            "denominator": len(application_rows),
            "rate": (
                len(pre_gate) / float(len(application_rows))
                if application_rows
                else None
            ),
            "scenario_ids": [r.get("scenario_id") for r in pre_gate],
            "note": "Pre-generation blocking is not LLM grounding success.",
        },
        "post_generation_operational_gates": {
            "numerator": len(post_gate),
            "denominator": len(application_rows),
            "rate": (
                len(post_gate) / float(len(application_rows))
                if application_rows
                else None
            ),
            "scenario_ids": [r.get("scenario_id") for r in post_gate],
            "note": (
                "A01 ask paths that never reach the LLM cannot demonstrate "
                "post-generation blocking."
            ),
        },
        "unsafe_escaped_to_final_application_answer": {
            **fraction(len(escaped_unsafe), len(application_rows)),
            "caveat": (
                "Smoke-set observation only. Not comprehensive prompt-injection "
                "resistance. Do not overgeneralize from n application rows."
            ),
        },
        "model_only_injection_follow": fraction(
            len(model_only_injection), len(model_only)
        ),
        "adversarial_application_rows": len(adversarial_app),
        "latency": {
            "full_request_all_application_rows_ms": _latency_stats(full_request_latencies),
            "full_request_llm_called_rows_only_ms": _latency_stats(
                generation_request_latencies
            ),
            "inference_ollama_or_llm_wall_ms": _latency_stats(inference_latencies),
            "pre_generation_blocked_rows_ms": _latency_stats(pre_gate_latencies),
            "note": (
                "Do not mix pre-gated (~0 ms) rows into inference latency. "
                "Use inference_ollama_or_llm_wall_ms / full_request_llm_called_rows_only_ms "
                "for generation timing."
            ),
        },
        # Back-compat aliases used by older analysis templates / tests.
        "n_rows": len(application_rows),
        "supported_structural_answer": fraction(
            len(supported_structural), len(supported)
        ),
        "supported_needs_human_factual_review": fraction(
            len(supported_needs_review), len(supported)
        ),
        "unsupported_abstention_accuracy": fraction(
            len(unsupported_ok), len(unsupported)
        ),
        "operational_policy_application_pass": fraction(len(policy_ok), len(policy)),
        "llm_generation_calls": fraction(len(llm_app_calls), len(application_rows)),
        "unsafe_or_injection_escaped_to_answer": fraction(
            len(escaped_unsafe), len(application_rows)
        ),
        "adversarial_rows": len(adversarial_app),
        "counts_by_actual_behavior": {
            ACTUAL_ANSWER: sum(
                1 for r in application_rows if r.get("actual_behavior") == ACTUAL_ANSWER
            ),
            ACTUAL_ABSTAIN_INSUFFICIENT: sum(
                1
                for r in application_rows
                if r.get("actual_behavior") == ACTUAL_ABSTAIN_INSUFFICIENT
            ),
            ACTUAL_ABSTAIN_OPERATIONAL: sum(
                1
                for r in application_rows
                if r.get("actual_behavior") == ACTUAL_ABSTAIN_OPERATIONAL
            ),
        },
        "major_failure_examples": failures,
        "caveat": (
            "n<=8 smoke set; do not overgeneralize. Structural pass ≠ factual "
            "correctness. Pre-generation policy blocking is not LLM grounding success. "
            "0 unsafe escapes on this smoke is not comprehensive injection resistance."
        ),
    }


def build_analysis_markdown(
    *,
    selection: dict[str, Any],
    summary: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    offline_tests_passed: bool | None = None,
    model_only_rows: Sequence[dict[str, Any]] | None = None,
) -> str:
    """Produce a concise comparison report with explicit n/d fractions."""

    def fmt(frac: dict[str, Any] | None) -> str:
        if not frac:
            return "n/a"
        num = frac.get("numerator")
        den = frac.get("denominator")
        rate = frac.get("rate")
        if den == 0 or rate is None:
            return f"{num}/{den} (undefined rate)"
        return f"{num}/{den} ({rate:.0%})"

    pops = summary.get("populations") or {}
    lat = summary.get("latency") or {}

    lines = [
        "# Phase 5B.2B — Grounded Answer Evaluation (smoke)",
        "",
        f"Selection: `{selection.get('selection_id')}` "
        f"(parent `{selection.get('parent_dataset_id')}` "
        f"v{selection.get('parent_version')}).",
        "",
        "## Denominators",
        "",
        f"- Selected scenario IDs: {pops.get('selected_scenarios', 8)}",
        f"- Application-level rows: {pops.get('application_rows')} "
        f"({pops.get('application_rows_note')})",
        f"- Model-only rows: {pops.get('model_only_rows')}",
        f"- Application rows that reached LLM: "
        f"{pops.get('llm_called_application_rows')}",
        f"- Pre-generation blocked application rows: "
        f"{pops.get('pre_generation_blocked_rows')}",
        f"- Post-generation blocked application rows: "
        f"{pops.get('post_generation_blocked_rows')}",
        "",
        "## Selection rationale",
        "",
    ]
    for item in selection.get("rationale") or []:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Separated outcome classes",
            "",
            "### Structural validity (supported)",
            f"- {fmt(summary.get('structural_validity_supported'))}",
            "",
            "### Citation page overlap (among answered supported)",
            f"- {fmt(summary.get('citation_page_overlap_among_answers'))}",
            "",
            "### Claim-level factual support",
            f"- {(summary.get('claim_level_factual_support') or {}).get('status')}: "
            f"{fmt((summary.get('claim_level_factual_support') or {}).get('supported_needing_review'))}",
            f"- {(summary.get('claim_level_factual_support') or {}).get('note')}",
            "",
            "### Correct abstention (unsupported / excluded)",
            f"- Application-level: {fmt(summary.get('unsupported_abstention_application'))}",
            "- Inspect per-row `model_output` to separate Qwen insufficient_evidence "
            "from malformed/invalid-citation fail-closed paths.",
            "",
            "### Operational-policy blocking",
            f"- Pre-generation policy rows: "
            f"{fmt(summary.get('operational_policy_pre_generation'))}",
            f"- Pre-generation gates among application rows: "
            f"{fmt(summary.get('pre_generation_operational_gates'))}",
            f"- Post-generation gates among application rows: "
            f"{fmt(summary.get('post_generation_operational_gates'))}",
            "",
            "### Prompt-injection outcomes",
            f"- Model-only injection follow: "
            f"{fmt(summary.get('model_only_injection_follow'))}",
            f"- Unsafe escaped to final application answer (smoke): "
            f"{fmt(summary.get('unsafe_escaped_to_final_application_answer'))}",
            f"- Caveat: "
            f"{(summary.get('unsafe_escaped_to_final_application_answer') or {}).get('caveat')}",
            "",
            "## Latency (separated populations)",
            "",
            f"- Full request, all application rows: {lat.get('full_request_all_application_rows_ms')}",
            f"- Full request, LLM-called rows only: {lat.get('full_request_llm_called_rows_only_ms')}",
            f"- Inference (Ollama total / llm_wall): {lat.get('inference_ollama_or_llm_wall_ms')}",
            f"- Pre-generation blocked rows: {lat.get('pre_generation_blocked_rows_ms')}",
            f"- {lat.get('note')}",
            "",
        ]
    )

    if offline_tests_passed is not None:
        lines.extend(
            [
                "## Offline tests",
                "",
                f"- Complete offline pytest suite: "
                f"{'PASSED' if offline_tests_passed else 'FAILED'}",
                "",
            ]
        )

    lines.extend(["## Application-level per-scenario rows", ""])
    for row in rows:
        lines.append(
            f"- **{row.get('scenario_id')}** ({row.get('category')}, "
            f"mode={row.get('eval_mode')}): expected=`{row.get('expected_behavior')}`, "
            f"actual=`{row.get('actual_behavior')}`, "
            f"app_pass={row.get('application_pass')}, "
            f"llm_called={row.get('llm_called')}, "
            f"gate={row.get('operational_gate')}, "
            f"needs_human_review={row.get('needs_human_review')}, "
            f"latency_ms={row.get('latency_ms')}"
        )
        for note in row.get("notes") or []:
            lines.append(f"  - note: {note}")

    if model_only_rows:
        lines.extend(["", "## Model-only rows (no application gates)", ""])
        for raw in model_only_rows:
            model = raw.get("raw_model") or {}
            lines.append(
                f"- **{raw.get('scenario_id')}** mode=`{raw.get('eval_mode')}`: "
                f"json_valid={raw.get('json_schema_valid')}, "
                f"followed_injection={raw.get('followed_injection_markers')}, "
                f"latency_ms={raw.get('latency_ms')}, "
                f"answer_preview={(model.get('model_answer_text') or '')[:120]!r}"
            )

    failures = summary.get("major_failure_examples") or []
    lines.extend(["", "## Major failure examples", ""])
    if not failures:
        lines.append("- None on automatic structural checks for application rows.")
    else:
        for item in failures:
            lines.append(
                f"- {item.get('scenario_id')}: actual={item.get('actual_behavior')}, "
                f"error={item.get('error')}, notes={item.get('notes')}"
            )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "- Keep retrieval and answer metrics separate; Hit@1≠factual support.",
            "- Require human review for every supported answer before claiming accuracy.",
            "- Expand unsupported/abstention cases before tuning generation prompts.",
            "- Keep adversarial fixtures off the production index; report raw vs app.",
            "- Do not treat pre-generation operational blocks as grounding wins.",
            "- Do not present smoke-set zero escapes as comprehensive injection resistance.",
            "",
            "## Limitations",
            "",
            f"- {summary.get('caveat')}",
            "- Eight scenarios only; not a production readiness gate.",
            "- No long copyrighted manual passages are embedded in this report.",
            "",
        ]
    )
    return "\n".join(lines)
