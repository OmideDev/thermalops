"""Load and validate the curated Greenheck ERV evaluation dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_DATASET_PATH = DATASETS_DIR / "greenheck_erv_eval_v1.json"

CATEGORIES = frozenset(
    {
        "supported_informational",
        "unsupported_absent",
        "unsupported_excluded_page",
        "operational_policy",
        "adversarial_injection",
    }
)

EXPECTED_BEHAVIORS = frozenset(
    {
        "answer_from_evidence",
        "abstain_insufficient_evidence",
        "abstain_operational_safety",
        "adversarial_measure_separately",
    }
)

EVAL_SPLITS = frozenset(
    {
        "retrieval_positive",
        "retrieval_negative",
        "policy",
        "adversarial",
    }
)

INSUFFICIENCY_KINDS = frozenset(
    {
        "absent_from_index",
        "excluded_page_31",
        "operational_policy",
        None,
    }
)

DIFFICULTIES = frozenset(
    {
        "direct_terminology",
        "natural_paraphrase",
        "indirect_reference",
        "multi_evidence",
    }
)


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_DATASET_PATH
    raw = json.loads(target.read_text(encoding="utf-8"))
    validate_dataset(raw)
    return raw


def validate_dataset(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("dataset root must be an object")
    required_meta = (
        "dataset_id",
        "document_id",
        "content_sha256",
        "allowed_pages",
        "excluded_pages",
        "embedding_model",
        "scenarios",
    )
    missing = [key for key in required_meta if key not in payload]
    if missing:
        raise ValueError(f"dataset missing field(s): {', '.join(missing)}")

    scenarios = payload["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a non-empty list")

    seen_ids: set[str] = set()
    for index, item in enumerate(scenarios):
        _validate_scenario(item, index=index, seen_ids=seen_ids)


def _validate_scenario(
    item: Any,
    *,
    index: int,
    seen_ids: set[str],
) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"scenario[{index}] must be an object")
    required = (
        "id",
        "category",
        "split",
        "question",
        "expected_behavior",
        "expected_pages",
        "expected_section_headings",
        "expected_chunk_ids",
        "reference_facts_paraphrased",
        "forbidden_claims",
        "insufficiency_kind",
        "rationale",
        "provenance",
    )
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(
            f"scenario[{index}] missing field(s): {', '.join(missing)}"
        )

    scenario_id = item["id"]
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError(f"scenario[{index}].id must be a non-empty string")
    if scenario_id in seen_ids:
        raise ValueError(f"duplicate scenario id: {scenario_id}")
    seen_ids.add(scenario_id)

    if item["category"] not in CATEGORIES:
        raise ValueError(f"{scenario_id}: invalid category")
    if item["split"] not in EVAL_SPLITS:
        raise ValueError(f"{scenario_id}: invalid split")
    if item["expected_behavior"] not in EXPECTED_BEHAVIORS:
        raise ValueError(f"{scenario_id}: invalid expected_behavior")
    if item["insufficiency_kind"] not in INSUFFICIENCY_KINDS:
        raise ValueError(f"{scenario_id}: invalid insufficiency_kind")

    if not isinstance(item["question"], str) or not item["question"].strip():
        raise ValueError(f"{scenario_id}: question must be non-empty")

    for field_name in (
        "expected_pages",
        "expected_section_headings",
        "expected_chunk_ids",
        "reference_facts_paraphrased",
        "forbidden_claims",
    ):
        if not isinstance(item[field_name], list):
            raise ValueError(f"{scenario_id}: {field_name} must be a list")

    if item["category"] == "supported_informational":
        if item["expected_behavior"] != "answer_from_evidence":
            raise ValueError(f"{scenario_id}: supported must answer_from_evidence")
        if not item["expected_chunk_ids"]:
            raise ValueError(f"{scenario_id}: supported needs expected_chunk_ids")
        if not item["expected_pages"]:
            raise ValueError(f"{scenario_id}: supported needs expected_pages")
        if item["insufficiency_kind"] is not None:
            raise ValueError(f"{scenario_id}: supported insufficiency_kind must be null")
        if item["split"] != "retrieval_positive":
            raise ValueError(f"{scenario_id}: supported split must be retrieval_positive")
        difficulty = item.get("difficulty")
        if difficulty is not None and difficulty not in DIFFICULTIES:
            raise ValueError(f"{scenario_id}: invalid difficulty")

    if item["category"] in {
        "unsupported_absent",
        "unsupported_excluded_page",
    }:
        if item["expected_chunk_ids"]:
            raise ValueError(f"{scenario_id}: negative cases must not set gold chunks")
        if item["expected_behavior"] != "abstain_insufficient_evidence":
            raise ValueError(
                f"{scenario_id}: unsupported must abstain_insufficient_evidence"
            )
        if item["split"] != "retrieval_negative":
            raise ValueError(f"{scenario_id}: unsupported split must be retrieval_negative")

    if item["category"] == "operational_policy":
        if item["expected_behavior"] != "abstain_operational_safety":
            raise ValueError(f"{scenario_id}: policy must abstain_operational_safety")
        if item["insufficiency_kind"] != "operational_policy":
            raise ValueError(f"{scenario_id}: policy insufficiency_kind mismatch")
        if item["expected_chunk_ids"]:
            raise ValueError(f"{scenario_id}: policy cases have no gold retrieval IDs")
        if item["split"] != "policy":
            raise ValueError(f"{scenario_id}: policy split must be policy")

    if item["category"] == "adversarial_injection":
        if item["expected_behavior"] != "adversarial_measure_separately":
            raise ValueError(f"{scenario_id}: adversarial behavior mismatch")
        if item["split"] != "adversarial":
            raise ValueError(f"{scenario_id}: adversarial split mismatch")
        if item["expected_chunk_ids"]:
            raise ValueError(
                f"{scenario_id}: adversarial must not use production gold chunks"
            )

    provenance = item["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError(f"{scenario_id}: provenance must be an object")
    for key in ("document_id", "content_sha256", "verification_method"):
        if key not in provenance:
            raise ValueError(f"{scenario_id}: provenance missing {key}")
