"""Load the Phase 5B.2B eight-scenario answer-evaluation selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.rag_eval.dataset import load_dataset

SELECTION_PATH = Path(__file__).resolve().parent / "answer_selection_v1.json"


def load_answer_selection(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else SELECTION_PATH
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("answer selection root must be an object")
    ids = payload.get("scenario_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("scenario_ids must be a non-empty list")
    if len(ids) > int(payload.get("max_scenarios") or 8):
        raise ValueError("selection exceeds max_scenarios")
    if len(set(ids)) != len(ids):
        raise ValueError("scenario_ids must be unique")
    for item in ids:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("scenario_ids must contain non-empty strings")
    return payload


def selected_scenarios(
    dataset: dict[str, Any] | None = None,
    *,
    selection: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return curated scenarios in selection order; preserve IDs and gold labels."""
    data = dataset if dataset is not None else load_dataset()
    sel = selection if selection is not None else load_answer_selection()
    by_id = {item["id"]: item for item in data["scenarios"]}
    missing = [sid for sid in sel["scenario_ids"] if sid not in by_id]
    if missing:
        raise ValueError(f"selection references unknown scenario id(s): {missing}")
    return [by_id[sid] for sid in sel["scenario_ids"]]
