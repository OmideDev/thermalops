"""
Structure-aware chunking for narrative HVAC manual text.

Builds retrieval-oriented chunks from already-normalized page text.
Does not invent headings, join table columns, or create symptom/cause/action
mappings. Flattened troubleshooting tables are flagged and excluded from
retrieval-ready narrative chunks.

Independent sections (reliably detected headings) are packed separately so
overlap never crosses a section boundary.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Initial heuristics for narrative manuals (not fixed production limits).
DEFAULT_TARGET_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 150

# Minimum non-heading body characters for ordinary narrative retention.
# Shorter bodies may still be kept when they look like safety/technical text.
MIN_INFORMATIVE_BODY_CHARS = 40

# Known flattened troubleshooting header triads (plain-text column labels).
# Comparison is case-insensitive; order matters (symptom/problem, cause, action).
_TABLE_HEADER_TRIADS = (
    ("Symptom", "Possible Cause", "Corrective Action"),
    ("Problem", "Cause", "Remedy"),
    ("Problem", "Possible Cause", "Corrective Action"),
    ("Fault", "Cause", "Corrective Action"),
    ("Fault", "Possible Cause", "Remedy"),
    ("Issue", "Cause", "Solution"),
)

# All known column-header labels (exact title case as written in manuals).
_TABLE_HEADER_LABELS = {part for triad in _TABLE_HEADER_TRIADS for part in triad}

# Short bodies that are still worth retrieving (warnings, lockout, fault codes).
_MEANINGFUL_SHORT_TECHNICAL = re.compile(
    r"(?is)^(?:"
    r"(?:WARNING|CAUTION|DANGER|NOTE|IMPORTANT)\b.*"
    r"|"
    r".*\b(?:disconnect|lock\s*out|tag\s*out|de-?energize|energize)\b.*"
    r"|"
    r".*\b(?:do not|must not|never)\b.*"
    r"|"
    r".*\b(?:fault|alarm|error)\s*(?:code|signal)?\b.*"
    r"|"
    r".*\b(?:code|alarm)\s*[:#]?\s*[A-Z]{0,3}\d{1,4}\b.*"
    r")"
)

# Prefer not to cut immediately after these incomplete measurement tails.
_MEASUREMENT_TAIL = re.compile(
    r"(?:"
    r"\d[\d,]*\.?\d*"
    r"(?:\s*(?:°\s*[FC]|CFM|RPM|VAC|VAC\)|in\.?\s*wg|ohms?|%))?"
    r"|"
    r"\+\d+%\s*/\s*-\d+%"
    r")$",
    re.IGNORECASE,
)

_PROCEDURE_OR_WARNING_START = re.compile(
    r"^(?:"
    r"\d+\.\s+"
    r"|"
    r"(?:WARNING|CAUTION|NOTE|IMPORTANT)\b"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageInput:
    """One extracted page ready for chunking."""

    page_number: int
    text: str


@dataclass(frozen=True)
class _Section:
    """One narrative section with an optional detected heading."""

    heading: str | None
    text: str


def _stable_chunk_id(
    document_id: str,
    content_sha256: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    """
    Build a reproducible chunk id from source identity + page + index + text.

    Including a short hash of the chunk text keeps the id stable for the same
    bytes and makes accidental reordering easier to spot.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    sha_prefix = content_sha256[:12] if content_sha256 else "nosha"
    return f"{document_id}:p{page_number}:c{chunk_index}:{sha_prefix}:{digest}"


def _is_probable_heading(line: str) -> bool:
    """
    Conservatively detect a section heading.

    Only returns True for short, title-like lines. Never invents a heading
    that is not already present as its own line in the source text.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 70:
        return False
    if stripped.endswith((".", ":", ";", ",")):
        return False
    if ". " in stripped or "? " in stripped:
        return False
    if stripped.isdigit():
        return False
    if stripped[:1].islower():
        return False
    if re.match(r"^\d+\.\s+", stripped):
        return False
    if stripped in _TABLE_HEADER_LABELS:
        return False
    if _normalize_header_label(stripped) in {
        _normalize_header_label(label) for label in _TABLE_HEADER_LABELS
    }:
        return False

    words = re.split(r"[\s/]+", stripped.replace("–", " ").replace("—", " "))
    words = [word for word in words if word]
    if not words or len(words) > 8:
        return False

    letter_count = sum(ch.isalpha() for ch in stripped)
    if letter_count < 3:
        return False

    titleish = 0
    for word in words:
        core = word.strip("()[]")
        if not core:
            continue
        if core.isupper() and any(ch.isalpha() for ch in core):
            titleish += 1
        elif core[0].isupper():
            titleish += 1
        elif re.match(r"^[A-Za-z]+\d+$", core):
            titleish += 1
    return titleish >= max(1, len(words) - 1)


def _previous_nonempty(lines: list[str], index: int) -> str | None:
    for cursor in range(index - 1, -1, -1):
        if lines[cursor].strip():
            return lines[cursor].strip()
    return None


def _starts_new_section(line: str, previous: str | None) -> bool:
    """
    True when `line` reliably opens a new independent section.

    Conservative: uncertain lines stay in the current section.
    """
    if not _is_probable_heading(line):
        return False
    if previous is None:
        return True
    if _is_probable_heading(previous):
        # Stacked titles at the top of a section (product name, then topic).
        return True
    if previous.endswith((".", "?", "!")):
        return True
    # A blank line before the heading is represented by walking nonempty
    # neighbors only; callers pass the previous nonempty line. Treat a short
    # previous heading-like gap via explicit blank handling in the splitter.
    return False


def _normalize_header_label(line: str) -> str:
    return " ".join(line.strip().lower().split())


def _find_flattened_table_span(
    lines: list[str],
) -> tuple[int, int, str] | None:
    """
    Return ``(start, end, reason)`` for a flattened troubleshooting table.

    Looks for known header triads on consecutive non-empty lines. When found,
    the unsafe region runs from the triad through the end of the page because
    plain text cannot recover column relationships.
    """
    nonempty_indexes = [i for i, line in enumerate(lines) if line.strip()]
    normalized_triads = [
        tuple(_normalize_header_label(part) for part in triad)
        for triad in _TABLE_HEADER_TRIADS
    ]
    for position, line_index in enumerate(nonempty_indexes[:-2]):
        candidate = (
            _normalize_header_label(lines[line_index]),
            _normalize_header_label(lines[nonempty_indexes[position + 1]]),
            _normalize_header_label(lines[nonempty_indexes[position + 2]]),
        )
        if candidate in normalized_triads:
            return line_index, len(lines) - 1, "flattened_troubleshooting_table"
    return None


def _is_troubleshooting_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    return "troubleshoot" in lowered or lowered.endswith("alarms")


def _line_looks_tableish(line: str) -> bool:
    """Heuristic: short/fragmented lines common in flattened table cells."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) <= 48:
        return True
    if stripped.endswith(".") and len(stripped) <= 70:
        return True
    return False


def _find_ambiguous_table_span(lines: list[str]) -> tuple[int, int, str] | None:
    """
    Conservatively flag dense table-like regions without inventing mappings.

    Covers false-negative risks such as:
    - different/unknown header names
    - continuation pages without repeated headers
    - troubleshooting sections dominated by short cell fragments

    Returns a span to exclude from retrieval-ready narrative chunks.
    """
    nonempty = [(i, line.strip()) for i, line in enumerate(lines) if line.strip()]
    if len(nonempty) < 12:
        return None

    # Dense short-line pages (often table continuations without headers).
    tableish_count = sum(1 for _, text in nonempty if _line_looks_tableish(text))
    tableish_ratio = tableish_count / len(nonempty)
    sentence_lines = sum(1 for _, text in nonempty if len(text) > 80)
    if tableish_ratio >= 0.7 and sentence_lines <= 3 and len(nonempty) >= 18:
        start = nonempty[0][0]
        # Keep a short product/title preamble if present before dense cells.
        for index, text in nonempty[:6]:
            if _is_probable_heading(text) or text.isdigit():
                continue
            start = index
            break
        return start, len(lines) - 1, "ambiguous_table_like_region"

    # Troubleshooting heading followed by dense short fragments, no triad.
    for position, (line_index, text) in enumerate(nonempty):
        if not _is_troubleshooting_heading(text):
            continue
        following = nonempty[position + 1 : position + 1 + 24]
        if len(following) < 10:
            continue
        follow_tableish = sum(1 for _, item in following if _line_looks_tableish(item))
        if follow_tableish / len(following) >= 0.65:
            # Start at first fragment after the troubleshooting title block.
            start = following[0][0]
            return start, len(lines) - 1, "ambiguous_table_like_region"

    return None


def _excluded_region(
    *,
    page_number: int,
    reason: str,
    start: int,
    end: int,
    lines: list[str],
) -> dict:
    excluded_lines = lines[start : end + 1]
    preview = "\n".join(excluded_lines[:12])
    if len(excluded_lines) > 12:
        preview += "\n..."
    notes = {
        "flattened_troubleshooting_table": (
            "Plain-text extraction lost column relationships. "
            "Do not index as narrative chunks. Later work should use "
            "PyMuPDF blocks/tables or layout-aware extraction."
        ),
        "ambiguous_table_like_region": (
            "Dense short-line region looks table-like but headers/columns "
            "are uncertain (alternate headers, continuation page, or missing "
            "repeated headers). Excluded from retrieval; keep for structured "
            "extraction review."
        ),
    }
    return {
        "page_number": page_number,
        "reason": reason,
        "retrieval_eligibility": "excluded",
        "safe_for_retrieval": False,
        "start_line": start + 1,
        "end_line": end + 1,
        "preview": preview,
        "text": "\n".join(excluded_lines),
        "note": notes.get(reason, "Excluded from retrieval-ready narrative chunks."),
    }


def _split_safe_and_excluded(
    text: str,
    page_number: int,
) -> tuple[str, list[dict]]:
    """
    Split page text into narrative-safe text and excluded table regions.

    Excluded regions keep full source text for later structured extraction.
    No symptom/cause/action mappings are invented.
    """
    if not text.strip():
        return "", []

    lines = text.split("\n")
    span = _find_flattened_table_span(lines)
    if span is None:
        span = _find_ambiguous_table_span(lines)
    if span is None:
        return text, []

    start, end, reason = span
    excluded = [_excluded_region(
        page_number=page_number,
        reason=reason,
        start=start,
        end=end,
        lines=lines,
    )]
    safe_lines = lines[:start]
    safe_text = "\n".join(safe_lines).rstrip()
    return safe_text, excluded


def _split_into_sections(text: str) -> list[_Section]:
    """
    Split narrative text into sections at reliably detected headings.

    Heading lines that begin a new section are not packed with the previous
    section's body. Stacked opening titles belong to the same new section;
    the last title becomes that section's heading.
    """
    if not text.strip():
        return []

    lines = text.split("\n")
    sections: list[_Section] = []
    current_lines: list[str] = []
    current_heading: str | None = None
    opening_titles: list[str] = []
    seen_body = False

    def flush() -> None:
        nonlocal current_lines, current_heading, opening_titles, seen_body
        body = "\n".join(current_lines).strip("\n")
        if not body.strip() and current_heading is None and not opening_titles:
            current_lines = []
            opening_titles = []
            seen_body = False
            return
        heading = current_heading
        if heading is None and opening_titles:
            heading = opening_titles[-1]
        sections.append(_Section(heading=heading, text=body))
        current_lines = []
        current_heading = None
        opening_titles = []
        seen_body = False

    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        previous = _previous_nonempty(lines, index)

        # Blank line: keep structure inside the current section.
        if not stripped:
            if current_lines or opening_titles or current_heading:
                current_lines.append(raw_line)
            continue

        # Skip lone page-number lines at the very start of a section buffer.
        if stripped.isdigit() and not seen_body and not current_lines:
            current_lines.append(raw_line)
            continue

        starts = _starts_new_section(stripped, previous)
        # Also start a section when a heading follows an explicit blank line
        # after body content (common in normalized manuals).
        if (
            not starts
            and _is_probable_heading(stripped)
            and seen_body
            and index > 0
            and not lines[index - 1].strip()
        ):
            starts = True

        if starts and (seen_body or (current_lines and current_heading)):
            # Close the previous section before starting a new independent one.
            # Keep stacked titles (no body yet) in the same opening block.
            if seen_body:
                flush()
                opening_titles = [stripped]
                current_heading = stripped
                current_lines = [raw_line]
                seen_body = False
                continue

        if _is_probable_heading(stripped) and not seen_body:
            opening_titles.append(stripped)
            current_heading = stripped
            current_lines.append(raw_line)
            continue

        current_lines.append(raw_line)
        seen_body = True

    flush()
    return sections


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines; keep single newlines inside a paragraph."""
    if not text.strip():
        return []
    parts = re.split(r"\n\s*\n", text)
    return [part.strip("\n") for part in parts if part.strip()]


def _looks_like_bad_cut(prefix: str) -> bool:
    """True when ending `prefix` mid-measurement or mid-numbered step is likely."""
    tail = prefix.rstrip()
    if not tail:
        return False
    last_line = tail.split("\n")[-1]
    if _MEASUREMENT_TAIL.search(last_line):
        if last_line.count("(") > last_line.count(")"):
            return True
    if _PROCEDURE_OR_WARNING_START.match(last_line) and len(last_line) < 40:
        return True
    if re.search(r"\d+\.\d+$", last_line) and not re.search(
        r"(?:°|[A-Za-z])$", last_line
    ):
        return True
    return False


def _overlap_prefix(text: str, overlap_chars: int) -> str:
    """Take a trailing overlap window, preferring a newline/space boundary."""
    if overlap_chars <= 0 or not text:
        return ""
    window = text[-overlap_chars:]
    newline_at = window.find("\n")
    if newline_at >= 0 and newline_at + 1 < len(window):
        window = window[newline_at + 1 :]
    else:
        space_at = window.find(" ")
        if space_at >= 0 and space_at + 1 < len(window):
            window = window[space_at + 1 :]
    return window.strip("\n")


def _hard_wrap(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Final fallback: wrap on spaces near the target size."""
    words = text.split(" ")
    pieces: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= target_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
            overlap = _overlap_prefix(current, overlap_chars)
            current = f"{overlap} {word}".strip() if overlap else word
        else:
            pieces.append(word)
            current = ""
    if current:
        pieces.append(current)
    return pieces


def _split_long_block(
    block: str,
    target_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split an oversized paragraph on newlines, then spaces, with overlap."""
    lines = block.split("\n")
    pieces: list[str] = []
    current = ""

    for line in lines:
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= target_chars:
            current = candidate
            continue
        if current:
            if _looks_like_bad_cut(current) and len(candidate) <= target_chars + 80:
                current = candidate
                continue
            pieces.append(current)
            overlap = _overlap_prefix(current, overlap_chars)
            current = f"{overlap}\n{line}".strip("\n") if overlap else line
            if len(current) > target_chars:
                pieces.extend(_hard_wrap(current, target_chars, overlap_chars))
                current = ""
        else:
            pieces.extend(_hard_wrap(line, target_chars, overlap_chars))
            current = ""

    if current.strip():
        pieces.append(current.strip("\n"))
    return pieces


def _merge_leading_heading_paragraphs(paragraphs: list[str]) -> list[str]:
    """
    Keep section titles attached to the following body paragraph.

    Otherwise a lone heading paragraph can flush as its own tiny chunk and be
    dropped as non-informative, leaving later chunks without the title line.
    """
    if len(paragraphs) < 2:
        return paragraphs

    merged: list[str] = []
    index = 0
    while index < len(paragraphs):
        paragraph = paragraphs[index]
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        only_headings = bool(lines) and all(_is_probable_heading(line) for line in lines)
        if only_headings and index + 1 < len(paragraphs):
            merged.append(f"{paragraph}\n\n{paragraphs[index + 1]}")
            index += 2
            continue
        merged.append(paragraph)
        index += 1
    return merged


def _pack_paragraphs(
    paragraphs: list[str],
    *,
    target_chars: int,
    overlap_chars: int,
) -> list[str]:
    """
    Pack paragraphs into chunks near `target_chars` with optional overlap.

    Caller must already confine `paragraphs` to a single section so overlap
    cannot cross independent section boundaries.
    """
    paragraphs = _merge_leading_heading_paragraphs(paragraphs)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip("\n"))
        current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= target_chars:
            current = candidate
            continue

        if not current:
            chunks.extend(
                _split_long_block(paragraph, target_chars, overlap_chars)
            )
            continue

        flush()
        if len(paragraph) <= target_chars:
            if chunks and overlap_chars > 0:
                overlap = _overlap_prefix(chunks[-1], overlap_chars)
                if overlap and not paragraph.startswith(overlap):
                    current = f"{overlap}\n\n{paragraph}"
                    if len(current) > target_chars + overlap_chars:
                        current = paragraph
                else:
                    current = paragraph
            else:
                current = paragraph
        else:
            chunks.extend(
                _split_long_block(paragraph, target_chars, overlap_chars)
            )

    flush()
    return chunks


def _informative_body(text: str) -> str:
    """Return chunk text with leading page numbers and heading lines removed."""
    lines = text.split("\n")
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index < len(lines) and lines[index].strip().isdigit():
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    while index < len(lines) and _is_probable_heading(lines[index]):
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
    return "\n".join(lines[index:]).strip()


def _looks_like_meaningful_short_technical(body: str) -> bool:
    """
    Keep short bodies that are clearly safety/fault/technical instructions.

    Conservative: unknown short strings are not retained just for being short.
    """
    compact = " ".join(body.split())
    if not compact:
        return False
    return _MEANINGFUL_SHORT_TECHNICAL.search(compact) is not None


def _is_heading_only_or_non_informative(text: str) -> bool:
    """
    True when a chunk should not be retrieval-ready.

    - Pure heading/page-number content → exclude
    - Ordinary short bodies (< MIN_INFORMATIVE_BODY_CHARS) → exclude
    - Short safety/fault/technical instructions → keep
    """
    body = _informative_body(text)
    if not body:
        return True
    if len(body) >= MIN_INFORMATIVE_BODY_CHARS:
        return False
    if _looks_like_meaningful_short_technical(body):
        return False
    return True


def chunk_page(
    page: PageInput,
    *,
    document_id: str,
    content_sha256: str,
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> tuple[list[dict], list[dict]]:
    """
    Chunk one page into retrieval-ready narrative pieces.

    Returns ``(chunks, excluded_regions)``.
    """
    if page.page_number < 1:
        raise ValueError("page_number must be 1-based and >= 1")

    safe_text, excluded = _split_safe_and_excluded(page.text, page.page_number)
    if not safe_text.strip():
        return [], excluded

    sections = _split_into_sections(safe_text)
    chunks: list[dict] = []
    chunk_index = 0

    for section in sections:
        packed = _pack_paragraphs(
            _paragraphs(section.text),
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
        for chunk_text in packed:
            if _is_heading_only_or_non_informative(chunk_text):
                excluded.append(
                    {
                        "page_number": page.page_number,
                        "reason": "heading_only_or_non_informative",
                        "retrieval_eligibility": "excluded",
                        "safe_for_retrieval": False,
                        "section_heading": section.heading,
                        "preview": chunk_text[:240],
                        "text": chunk_text,
                        "note": (
                            "Preserved for audit but excluded from retrieval "
                            "because the chunk has little or no body beyond "
                            "headings or page numbers."
                        ),
                    }
                )
                continue

            chunk = {
                "chunk_id": _stable_chunk_id(
                    document_id,
                    content_sha256,
                    page.page_number,
                    chunk_index,
                    chunk_text,
                ),
                "document_id": document_id,
                "content_sha256": content_sha256,
                "page_number": page.page_number,
                "chunk_index": chunk_index,
                "section_heading": section.heading,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "retrieval_eligibility": "eligible",
                "safe_for_retrieval": True,
            }
            chunks.append(chunk)
            chunk_index += 1

    return chunks, excluded


def chunk_pages(
    pages: list[PageInput],
    *,
    document_id: str,
    content_sha256: str,
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> dict:
    """
    Chunk many pages in deterministic page / chunk_index order.

    Empty or whitespace-only pages contribute no chunks.
    """
    all_chunks: list[dict] = []
    all_excluded: list[dict] = []

    for page in sorted(pages, key=lambda item: item.page_number):
        chunks, excluded = chunk_page(
            page,
            document_id=document_id,
            content_sha256=content_sha256,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
        all_chunks.extend(chunks)
        all_excluded.extend(excluded)

    return {
        "document_id": document_id,
        "content_sha256": content_sha256,
        "chunk_count": len(all_chunks),
        "chunks": all_chunks,
        "excluded_regions": all_excluded,
    }
