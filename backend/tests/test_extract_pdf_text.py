"""Focused tests for RAG Phase 1 PDF text extraction.

Uses tiny PDFs created in temporary directories so CI never depends on the
untracked Greenheck IOM manual.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pymupdf
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "extract_pdf_text.py"


def load_extract_module():
    """Load scripts/extract_pdf_text.py as a module without installing a package."""
    spec = importlib.util.spec_from_file_location("extract_pdf_text", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


extract_pdf_text = load_extract_module()


def make_sample_pdf(path: Path, page_texts: list[str]) -> Path:
    """Create a small multi-page PDF with known text on each page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    try:
        for text in page_texts:
            page = document.new_page()
            # Insert near the top-left so extraction is reliable.
            page.insert_text((72, 72), text)
        document.save(path)
    finally:
        document.close()
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    return make_sample_pdf(
        tmp_path / "sample_manual.pdf",
        [
            "PAGE_ONE_MARKER frost control",
            "PAGE_TWO_MARKER dirty filter sensor",
            "PAGE_THREE_MARKER troubleshooting table",
        ],
    )


def test_extracts_known_page_text(sample_pdf: Path):
    result = extract_pdf_text.extract_pages(sample_pdf, [2])

    assert result["requested_pages"] == [2]
    assert len(result["pages"]) == 1
    page = result["pages"][0]
    assert page["page_number"] == 2
    assert "PAGE_TWO_MARKER dirty filter sensor" in page["text"]
    assert page["is_blank"] is False


def test_one_based_to_zero_based_page_conversion(sample_pdf: Path):
    result = extract_pdf_text.extract_pages(sample_pdf, [1, 3])

    assert result["pages"][0]["page_number"] == 1
    assert result["pages"][0]["page_index"] == 0
    assert result["pages"][1]["page_number"] == 3
    assert result["pages"][1]["page_index"] == 2
    assert "PAGE_ONE_MARKER" in result["pages"][0]["text"]
    assert "PAGE_THREE_MARKER" in result["pages"][1]["text"]


def test_out_of_range_page_raises(sample_pdf: Path):
    with pytest.raises(ValueError, match="out of range"):
        extract_pdf_text.extract_pages(sample_pdf, [99])


def test_missing_pdf_raises(tmp_path: Path):
    missing = tmp_path / "does-not-exist.pdf"
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        extract_pdf_text.extract_pages(missing, [1])


def test_metadata_shape_is_portable(sample_pdf: Path):
    result = extract_pdf_text.extract_pages(sample_pdf, [1])
    metadata = result["metadata"]

    assert metadata["document_id"] == "sample_manual.pdf"
    assert metadata["file_name"] == "sample_manual.pdf"
    assert metadata["page_count"] == 3
    assert isinstance(metadata["content_sha256"], str)
    assert len(metadata["content_sha256"]) == 64
    assert "source_path" not in metadata
    # Temp dirs are outside backend/, so relative_path should be omitted.
    assert metadata["relative_path"] is None

    # Output shape for pages
    page = result["pages"][0]
    assert set(page) >= {
        "page_number",
        "page_index",
        "char_count",
        "line_count",
        "is_blank",
        "text",
    }


def test_relative_path_when_pdf_under_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Point BACKEND_ROOT at a temp tree so we never write into real manuals/.
    fake_backend = tmp_path / "backend"
    manuals_dir = fake_backend / "data" / "manuals"
    pdf_path = manuals_dir / "portable_manual.pdf"
    make_sample_pdf(pdf_path, ["BACKEND_RELATIVE_MARKER"])

    monkeypatch.setattr(extract_pdf_text, "BACKEND_ROOT", fake_backend)

    result = extract_pdf_text.extract_pages(pdf_path, [1])
    assert result["metadata"]["relative_path"] == "data/manuals/portable_manual.pdf"
    assert result["metadata"]["document_id"] == "portable_manual.pdf"


def test_utf8_json_output_roundtrip(sample_pdf: Path, tmp_path: Path):
    # JSON export must preserve Unicode (manuals often contain ligatures like ﬁ).
    # Inject the characters after extraction so the test does not depend on PDF
    # font embedding (Helvetica cannot encode U+FB01).
    result = extract_pdf_text.extract_pages(sample_pdf, [1])
    result["pages"][0]["text"] = "Dirty ﬁlter sensor — UTF8"
    output_path = tmp_path / "out" / "extraction.json"

    extract_pdf_text.write_json(result, output_path)

    raw = output_path.read_bytes()
    assert b"source_path" not in raw
    assert "ﬁ".encode("utf-8") in raw
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert "ﬁ" in loaded["pages"][0]["text"]
    assert "—" in loaded["pages"][0]["text"]
    assert loaded["metadata"]["document_id"] == "sample_manual.pdf"


def test_preview_cli_does_not_write_json(sample_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.chdir(tmp_path)
    unexpected = tmp_path / "extraction_sample.json"

    exit_code = extract_pdf_text.main(
        ["--pdf", str(sample_pdf), "--pages", "2", "--max-chars", "80"]
    )

    assert exit_code == 0
    assert not unexpected.exists()
    captured = capsys.readouterr()
    assert "PAGE_TWO_MARKER" in captured.out


def test_optional_json_export_via_output_flag(sample_pdf: Path, tmp_path: Path):
    output_path = tmp_path / "export.json"

    exit_code = extract_pdf_text.main(
        [
            "--pdf",
            str(sample_pdf),
            "--pages",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["pages"][0]["page_number"] == 1
    assert "PAGE_ONE_MARKER" in payload["pages"][0]["text"]
    assert "source_path" not in payload["metadata"]


def test_cli_missing_pdf_exit_code(tmp_path: Path):
    missing = tmp_path / "missing.pdf"
    exit_code = extract_pdf_text.main(["--pdf", str(missing), "--pages", "1"])
    assert exit_code == 1


def test_cli_out_of_range_exit_code(sample_pdf: Path):
    exit_code = extract_pdf_text.main(
        ["--pdf", str(sample_pdf), "--pages", "50"]
    )
    assert exit_code == 2
