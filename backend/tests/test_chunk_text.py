"""Focused tests for structure-aware RAG chunking (synthetic text only)."""

from __future__ import annotations

from app.rag.chunk import (
    DEFAULT_OVERLAP_CHARS,
    DEFAULT_TARGET_CHARS,
    PageInput,
    chunk_page,
    chunk_pages,
)


DOC_ID = "sample_manual.pdf"
DOC_SHA = "a" * 64


def _chunk(text: str, page: int = 1, **kwargs):
    return chunk_page(
        PageInput(page_number=page, text=text),
        document_id=DOC_ID,
        content_sha256=DOC_SHA,
        **kwargs,
    )


def test_short_text_single_chunk():
    text = "Frost Control\n\nKeep the wheel free of frost during winter."
    chunks, excluded = _chunk(text)
    assert excluded == []
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["safe_for_retrieval"] is True
    assert chunks[0]["section_heading"] == "Frost Control"


def test_empty_page_yields_no_chunks():
    chunks, excluded = _chunk("   \n\n  ")
    assert chunks == []
    assert excluded == []


def test_long_text_uses_paragraph_boundaries():
    paragraphs = [
        f"Section paragraph {index}. " + ("word " * 40)
        for index in range(6)
    ]
    text = "Overview\n\n" + "\n\n".join(paragraphs)
    chunks, excluded = _chunk(text, target_chars=400, overlap_chars=50)
    assert excluded == []
    assert len(chunks) >= 2
    assert any("\n\n" in chunk["text"] for chunk in chunks)


def test_chunk_overlap_within_same_section():
    first = "Alpha block. " + ("alpha " * 80)
    second = "Beta block. " + ("beta " * 80)
    text = f"Overview\n\n{first}\n\n{second}"
    chunks, _ = _chunk(text, target_chars=500, overlap_chars=80)
    assert len(chunks) >= 2
    assert "alpha" in chunks[1]["text"] or chunks[0]["text"][-40:] in chunks[1]["text"]


def test_deterministic_output_and_stable_ids():
    text = "Heading One\n\n" + ("Sentence about airflow. " * 60)
    first_chunks, _ = _chunk(text, target_chars=350, overlap_chars=40)
    second_chunks, _ = _chunk(text, target_chars=350, overlap_chars=40)
    assert first_chunks == second_chunks
    assert first_chunks[0]["chunk_id"] == second_chunks[0]["chunk_id"]
    assert first_chunks[0]["document_id"] == DOC_ID
    assert first_chunks[0]["content_sha256"] == DOC_SHA


def test_source_page_metadata_and_ordering():
    result = chunk_pages(
        [
            PageInput(
                page_number=3,
                text="Page Three\n\nContent on three describes damper travel and limits.",
            ),
            PageInput(
                page_number=1,
                text="Page One\n\nContent on one describes supply fan start interlocks.",
            ),
        ],
        document_id=DOC_ID,
        content_sha256=DOC_SHA,
    )
    pages = [chunk["page_number"] for chunk in result["chunks"]]
    assert pages == sorted(pages)
    assert result["chunks"][0]["page_number"] == 1
    assert result["chunks"][0]["chunk_index"] == 0


def test_technical_units_and_measurements_preserved():
    text = (
        "Setpoints\n\n"
        "Maintain -20 °F (-28.9 °C) and 1,250 CFM @ 0.75 in. wg.\n"
        "Voltage must stay within +10% / -15% of coil nominal."
    )
    chunks, excluded = _chunk(text)
    assert excluded == []
    joined = "\n".join(chunk["text"] for chunk in chunks)
    assert "-20 °F (-28.9 °C)" in joined
    assert "1,250 CFM @ 0.75 in. wg" in joined
    assert "+10% / -15%" in joined


def test_heading_preservation_when_detectable():
    text = "Electric Preheat\n\nUse the thermodisc to sense outdoor air temperature."
    chunks, _ = _chunk(text)
    assert chunks[0]["section_heading"] == "Electric Preheat"


def test_two_sections_on_same_page_stay_separate():
    text = (
        "Frost Control\n\n"
        "Use timed exhaust when outdoor air is extremely cold.\n"
        "Keep the energy wheel free of frost.\n\n"
        "CO2 Sensor\n\n"
        "This accessory is used in demand control ventilation applications.\n"
        "The sensor measures carbon dioxide concentration in the space."
    )
    chunks, excluded = _chunk(text, target_chars=1000, overlap_chars=150)
    assert excluded == []
    assert len(chunks) == 2
    assert chunks[0]["section_heading"] == "Frost Control"
    assert chunks[1]["section_heading"] == "CO2 Sensor"
    assert "CO2 Sensor" not in chunks[0]["text"]
    assert "Frost Control" not in chunks[1]["text"]
    assert "demand control ventilation" in chunks[1]["text"]


def test_no_overlap_across_independent_section_boundaries():
    frost_body = "Frost details. " + ("frost " * 60)
    co2_body = "Sensor details. " + ("sensor " * 60)
    text = f"Frost Control\n\n{frost_body}\n\nCO2 Sensor\n\n{co2_body}"
    chunks, _ = _chunk(text, target_chars=200, overlap_chars=80)
    frost_chunks = [c for c in chunks if c["section_heading"] == "Frost Control"]
    co2_chunks = [c for c in chunks if c["section_heading"] == "CO2 Sensor"]
    assert frost_chunks and co2_chunks
    # First CO2 chunk must not contain frost overlap tails from the prior section.
    assert "frost" not in co2_chunks[0]["text"].lower()
    assert "sensor" in co2_chunks[0]["text"].lower()


def test_heading_metadata_matches_section_body():
    text = (
        "Dirty Filter Sensor\n\n"
        "Dirty filter sensors monitor pressure drop across filters.\n\n"
        "Wheel Speed\n\n"
        "Wheel speed may be modulated during economizer operation."
    )
    chunks, _ = _chunk(text)
    by_heading = {chunk["section_heading"]: chunk["text"] for chunk in chunks}
    assert "pressure drop across filters" in by_heading["Dirty Filter Sensor"]
    assert "economizer operation" in by_heading["Wheel Speed"]
    assert "economizer" not in by_heading["Dirty Filter Sensor"]


def test_heading_only_region_excluded_but_preserved():
    text = "Troubleshooting – Unit"
    chunks, excluded = _chunk(text, page=31)
    assert chunks == []
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "heading_only_or_non_informative"
    assert excluded[0]["safe_for_retrieval"] is False
    assert "Troubleshooting" in excluded[0]["text"]


def test_informative_short_section_is_kept():
    text = (
        "Bypass Damper\n\n"
        "Open the bypass damper before service to relieve pressure safely."
    )
    chunks, excluded = _chunk(text)
    assert excluded == []
    assert len(chunks) == 1
    assert chunks[0]["section_heading"] == "Bypass Damper"
    assert "relieve pressure safely" in chunks[0]["text"]


def test_flattened_table_flagged_and_not_reconstructed():
    narrative = "Troubleshooting – Unit"
    table = (
        "Symptom\n"
        "Possible Cause\n"
        "Corrective Action\n"
        "Blower fails to\n"
        "operate\n"
        "Blown fuse or open circuit breaker.\n"
        "Replace fuse or reset circuit breaker and check amps."
    )
    text = f"{narrative}\n{table}"
    chunks, excluded = _chunk(text, page=31)

    reasons = {item["reason"] for item in excluded}
    assert "flattened_troubleshooting_table" in reasons
    assert "heading_only_or_non_informative" in reasons
    assert chunks == []

    table_region = next(
        item for item in excluded if item["reason"] == "flattened_troubleshooting_table"
    )
    assert "Blown fuse or open circuit breaker." in table_region["text"]
    assert "symptom" not in table_region
    assert "possible_cause" not in table_region


def test_short_warning_instruction_is_kept():
    text = "WARNING\n\nDisconnect power before service."
    chunks, excluded = _chunk(text)
    assert excluded == []
    assert len(chunks) == 1
    assert chunks[0]["safe_for_retrieval"] is True
    assert chunks[0]["retrieval_eligibility"] == "eligible"
    assert "Disconnect power before service." in chunks[0]["text"]


def test_short_fault_message_is_kept():
    text = "Alarm\n\nFault code F12 present."
    chunks, excluded = _chunk(text)
    assert excluded == []
    assert len(chunks) == 1
    assert "Fault code F12" in chunks[0]["text"]


def test_heading_only_still_excluded_after_short_tech_fix():
    text = "Troubleshooting – Unit"
    chunks, excluded = _chunk(text)
    assert chunks == []
    assert excluded[0]["reason"] == "heading_only_or_non_informative"


def test_short_non_technical_noise_still_excluded():
    text = "Overview\n\nSee above."
    chunks, excluded = _chunk(text)
    assert chunks == []
    assert any(item["reason"] == "heading_only_or_non_informative" for item in excluded)


def test_alternate_table_headers_are_excluded():
    text = (
        "Diagnostics\n"
        "Problem\n"
        "Cause\n"
        "Remedy\n"
        "Fan will not start\n"
        "Blown fuse\n"
        "Replace fuse"
    )
    chunks, excluded = _chunk(text, page=12)
    reasons = {item["reason"] for item in excluded}
    assert "flattened_troubleshooting_table" in reasons
    joined = "\n".join(chunk["text"] for chunk in chunks)
    assert "Blown fuse" not in joined


def test_ambiguous_table_continuation_without_headers_excluded():
    # Dense short fragments, no triad — resembles a table continuation page.
    rows = [
        "Blower fails to",
        "operate",
        "Blown fuse.",
        "Replace fuse.",
        "Motor overloaded.",
        "Reset VFD.",
        "Low airflow",
        "(cfm)",
        "Dirty filter.",
        "Replace filters.",
        "High airflow",
        "(cfm)",
        "Fan speed high.",
        "Reduce RPM.",
        "Excessive",
        "noise",
        "Wheel rubbing.",
        "Realign wheel.",
        "Low voltage.",
        "Check supply.",
    ]
    text = "Energy Recovery Ventilator\n" + "\n".join(rows)
    chunks, excluded = _chunk(text, page=32)
    assert any(item["reason"] == "ambiguous_table_like_region" for item in excluded)
    assert all(chunk["retrieval_eligibility"] == "eligible" for chunk in chunks)
    # Table fragments must not become retrieval chunks.
    retrieval = "\n".join(chunk["text"] for chunk in chunks)
    assert "Blown fuse." not in retrieval
    table = next(item for item in excluded if item["reason"] == "ambiguous_table_like_region")
    assert "Blown fuse." in table["text"]
    assert table["retrieval_eligibility"] == "excluded"


def test_defaults_match_documented_heuristics():
    assert DEFAULT_TARGET_CHARS == 1000
    assert DEFAULT_OVERLAP_CHARS == 150
