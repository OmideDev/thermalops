"""Offline schema validation for the curated RAG evaluation dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag_eval.dataset import (
    DEFAULT_DATASET_PATH,
    load_dataset,
    validate_dataset,
)


def test_default_dataset_file_exists():
    assert DEFAULT_DATASET_PATH.is_file()


def test_load_and_validate_default_dataset():
    dataset = load_dataset()
    assert dataset["dataset_id"] == "greenheck_erv_eval_v1"
    assert dataset["version"] == "1.0.1"
    assert dataset["allowed_pages"] == [15, 29, 33]
    assert 31 in dataset["excluded_pages"]
    counts = dataset["scenario_counts"]
    assert counts["total"] == len(dataset["scenarios"])
    assert counts["supported_informational"] == 14
    assert counts["unsupported_absent"] == 8
    assert counts["unsupported_excluded_page"] == 6
    assert counts["operational_policy"] == 4
    assert counts["adversarial_injection"] == 8
    supported = [
        s for s in dataset["scenarios"] if s["category"] == "supported_informational"
    ]
    assert all("difficulty" in s for s in supported)


def test_holdout_paraphrase_dataset_validates():
    holdout_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "rag_eval"
        / "datasets"
        / "greenheck_erv_eval_v1_holdout_paraphrase.json"
    )
    dataset = load_dataset(holdout_path)
    assert dataset["parent_dataset_id"] == "greenheck_erv_eval_v1"
    assert len(dataset["scenarios"]) == 4


def test_supported_scenarios_have_gold_chunks_on_allowed_pages():
    dataset = load_dataset()
    allowed = set(dataset["allowed_pages"])
    for scenario in dataset["scenarios"]:
        if scenario["category"] != "supported_informational":
            continue
        assert scenario["expected_chunk_ids"]
        assert set(scenario["expected_pages"]).issubset(allowed)
        assert 31 not in scenario["expected_pages"]


def test_negative_and_policy_have_no_gold_chunks():
    dataset = load_dataset()
    for scenario in dataset["scenarios"]:
        if scenario["category"] == "supported_informational":
            continue
        assert scenario["expected_chunk_ids"] == []


def test_validate_rejects_supported_without_gold():
    dataset = load_dataset()
    bad = json.loads(json.dumps(dataset))
    for scenario in bad["scenarios"]:
        if scenario["category"] == "supported_informational":
            scenario["expected_chunk_ids"] = []
            break
    with pytest.raises(ValueError, match="expected_chunk_ids"):
        validate_dataset(bad)


def test_adversarial_fixture_file_is_labeled_synthetic():
    path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "rag_eval"
        / "fixtures"
        / "adversarial_synthetic_passages.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "SYNTHETIC" in payload["note"].upper() or "synthetic" in payload["note"]
    assert payload["passages"]
    for passage in payload["passages"]:
        assert "SYNTHETIC" in passage["section_heading"]
