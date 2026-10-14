"""Focused tests for RAG text normalization (synthetic examples only)."""

from __future__ import annotations

from app.rag.normalize import normalize_text


def test_crlf_and_cr_become_lf():
    assert normalize_text("line1\r\nline2\rline3\n") == "line1\nline2\nline3\n"


def test_non_breaking_spaces_become_normal_spaces():
    raw = "Dirty\u00a0filter\u00a0sensor"
    assert normalize_text(raw) == "Dirty filter sensor"


def test_multiple_blank_lines_collapse_to_max_two():
    raw = "Frost Control\n\n\n\n\nTroubleshooting"
    assert normalize_text(raw) == "Frost Control\n\n\nTroubleshooting"

    # Exactly two blank lines must be preserved.
    two_blanks = "A\n\n\nB"
    assert normalize_text(two_blanks) == "A\n\n\nB"


def test_trailing_whitespace_removed_indentation_kept():
    raw = "  indented heading  \n\tstill indented\t\n"
    assert normalize_text(raw) == "  indented heading\n\tstill indented\n"


def test_unicode_preserved():
    raw = "Dirty ﬁlter sensor — check ΔT"
    assert normalize_text(raw) == "Dirty ﬁlter sensor — check ΔT"


def test_technical_measurements_preserved():
    raw = "Setpoint: -20 °F ( -28.9 °C )\nAirflow: 1,250 CFM @ 0.75 in. wg"
    assert normalize_text(raw) == raw


def test_empty_string():
    assert normalize_text("") == ""


def test_idempotent():
    samples = [
        "",
        "plain",
        "a\r\nb\rc\n",
        "x\u00a0y\u00a0z",
        "A\n\n\n\n\nB\n",
        "  keep\t  \n",
        "ﬁ — ΔT 12.5 °F\x00",
        "Symptom\n\nPossible Cause\n\n\n\nCorrective Action\n",
    ]
    for sample in samples:
        once = normalize_text(sample)
        twice = normalize_text(once)
        assert twice == once


def test_does_not_join_lines_or_drop_repeated_phrases():
    raw = "Blower fail\nBlower fail\nto start\nto start"
    assert normalize_text(raw) == raw


def test_nul_characters_removed():
    assert normalize_text("before\x00after") == "beforeafter"
