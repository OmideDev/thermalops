"""
Standalone PDF text extraction for ThermalOps RAG.

Extracts plain text page by page with PyMuPDF (`import pymupdf`).
Each page keeps raw ``text`` plus additive ``normalized_text``.
Does not call the database, embeddings, or FastAPI.

Usage (from the backend folder):

    # Console preview only (does NOT write any JSON file)
    python scripts/extract_pdf_text.py --pages 15 31

    # Print full JSON to stdout (UTF-8 safe; still does not write a file)
    python scripts/extract_pdf_text.py --pages 15 31 --json

    # Explicitly write UTF-8 JSON to a file
    python scripts/extract_pdf_text.py --pages 15 31 --output data/manuals/extraction_sample.json

    # Custom PDF path
    python scripts/extract_pdf_text.py --pdf data/manuals/460945ERT_iom.pdf --pages 15 31
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# PyMuPDF 1.28+ is imported as `pymupdf` (the old `import fitz` name is deprecated).
# Runtime import works; Pyrefly may still flag native package resolution.
import pymupdf  # pyrefly: ignore[missing-import]


# Resolve paths relative to the backend/ directory, not the caller's cwd.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = BACKEND_ROOT / "data" / "manuals" / "460945ERT_iom.pdf"

# Allow `from app.rag...` when this script is launched from backend/.
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.normalize import normalize_text


def sha256_file(path: Path) -> str:
    """Return a portable content fingerprint for the PDF bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_relative_path(pdf_path: Path) -> str | None:
    """
    Return a repo-portable path using forward slashes when the PDF lives
    under backend/. Otherwise return None (caller should rely on document_id).
    """
    try:
        relative = pdf_path.resolve().relative_to(BACKEND_ROOT.resolve())
    except ValueError:
        return None
    return relative.as_posix()


def build_metadata(document: pymupdf.Document, pdf_path: Path) -> dict:
    """
    Collect document-level facts useful later for RAG citations.

    Avoids absolute machine-specific paths. Identification is portable via
    document_id (file name), optional relative_path, and content_sha256.
    """
    meta = document.metadata or {}
    file_name = pdf_path.name
    return {
        "document_id": file_name,
        "file_name": file_name,
        "relative_path": portable_relative_path(pdf_path),
        "content_sha256": sha256_file(pdf_path),
        "page_count": document.page_count,
        "is_encrypted": document.is_encrypted,
        "is_pdf": document.is_pdf,
        "title": meta.get("title") or None,
        "author": meta.get("author") or None,
        "subject": meta.get("subject") or None,
        "keywords": meta.get("keywords") or None,
        "creator": meta.get("creator") or None,
        "producer": meta.get("producer") or None,
        "creation_date": meta.get("creationDate") or None,
        "mod_date": meta.get("modDate") or None,
        "format": meta.get("format") or None,
    }


def extract_page(document: pymupdf.Document, page_number: int) -> dict:
    """
    Extract text from one 1-based page number.

    Citations and CLI arguments use 1-based page numbers. PyMuPDF's API uses
    0-based indexes, so page 15 maps to index 14.
    """
    if page_number < 1 or page_number > document.page_count:
        raise ValueError(
            f"Page {page_number} is out of range "
            f"(document has {document.page_count} pages)."
        )

    page_index = page_number - 1
    page = document.load_page(page_index)
    # Keep raw extraction untouched; normalized copy is additive only.
    text = page.get_text("text")
    normalized = normalize_text(text)

    return {
        "page_number": page_number,
        "page_index": page_index,
        "char_count": len(text),
        "line_count": len(text.splitlines()),
        "is_blank": not text.strip(),
        "text": text,
        "normalized_text": normalized,
        "normalized_char_count": len(normalized),
    }


def extract_pages(pdf_path: Path, page_numbers: list[int]) -> dict:
    """Open the PDF once, extract requested pages, then close the file."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    # `with` ensures the PDF handle is closed even if an error occurs.
    with pymupdf.open(pdf_path) as document:
        metadata = build_metadata(document, pdf_path)
        pages: list[dict] = []

        for page_number in page_numbers:
            pages.append(extract_page(document, page_number))

        return {
            "metadata": metadata,
            "requested_pages": page_numbers,
            "pages": pages,
        }


def write_json(result: dict, output_path: Path) -> None:
    """Write extraction result as UTF-8 JSON (explicit export only)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text from an HVAC manual PDF page by page. "
            "Preview mode prints to the console only. "
            "Use --json to print JSON on stdout, and/or --output to write a UTF-8 JSON file."
        ),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Path to the PDF (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        default=[15, 31],
        help="1-based page numbers to extract (default: 15 31)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON on stdout (does not write a file by itself).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path to write UTF-8 JSON. "
            "Required when you want a file; preview mode never writes JSON automatically."
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2500,
        help="Max characters to preview per page in text mode (default: 2500).",
    )
    return parser.parse_args(argv)


def safe_print(text: str) -> None:
    """
    Print Unicode safely on Windows consoles that use a legacy code page.

    Some manuals contain ligatures (e.g. U+FB01 'ﬁ') that cp1256 cannot encode.
    """
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.buffer.write((text + "\n").encode(encoding, errors="replace"))
    sys.stdout.buffer.flush()


def print_preview(result: dict, max_chars: int) -> None:
    metadata = result["metadata"]
    safe_print("=== PDF metadata ===")
    for key, value in metadata.items():
        safe_print(f"{key}: {value}")

    safe_print("")
    for page in result["pages"]:
        safe_print("=" * 72)
        safe_print(
            f"Page {page['page_number']} "
            f"(index {page['page_index']}) | "
            f"{page['char_count']} chars | "
            f"{page['line_count']} lines | "
            f"blank={page['is_blank']}"
        )
        safe_print("-" * 72)
        text = page["text"]
        if len(text) > max_chars:
            safe_print(text[:max_chars])
            safe_print("")
            safe_print(
                f"...[truncated: showing first {max_chars} of {len(text)} chars]..."
            )
        else:
            safe_print(text)
        safe_print("")


def resolve_pdf_path(pdf_arg: Path) -> Path:
    """Resolve the CLI PDF path without requiring a specific working directory."""
    if pdf_arg.is_absolute():
        return pdf_arg

    cwd_candidate = (Path.cwd() / pdf_arg).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    backend_candidate = (BACKEND_ROOT / pdf_arg).resolve()
    if backend_candidate.exists():
        return backend_candidate

    # Fall back to cwd resolution so missing-file errors show a clear path.
    return cwd_candidate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pdf_path = resolve_pdf_path(args.pdf)

    try:
        result = extract_pages(pdf_path, args.pages)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # unexpected PDF/library failures
        print(f"ERROR: failed to extract text: {exc}", file=sys.stderr)
        return 3

    if args.output is not None:
        output_path = (
            args.output
            if args.output.is_absolute()
            else (Path.cwd() / args.output).resolve()
        )
        write_json(result, output_path)
        safe_print(f"Wrote UTF-8 JSON to: {output_path}")

    if args.json:
        safe_print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.output is None:
        # Preview-only mode: never write files.
        print_preview(result, args.max_chars)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
