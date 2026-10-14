"""
Deterministic plain-text normalization for RAG ingestion.

Operates on already-extracted PDF text. Does not paraphrase, join table
columns, or invent structure. Raw extraction output should be kept separately.
"""

from __future__ import annotations

import re

# Three or more consecutive blank lines appear as four or more newlines when
# blank lines are represented as empty strings between content lines.
_MULTI_BLANK_LINES = re.compile(r"\n{4,}")


def normalize_text(text: str) -> str:
    """
    Normalize extracted manual text for downstream RAG steps.

    Rules (applied in order):
    1. Remove NUL characters.
    2. Normalize Windows CRLF and legacy CR line endings to LF.
    3. Replace non-breaking spaces with normal spaces.
    4. Strip trailing whitespace on each line (leading indentation kept).
    5. Collapse runs of 3+ blank lines to at most 2 blank lines
       (implemented as regex ``\\n{4,}`` → ``\\n\\n\\n``).

    The function is deterministic and idempotent:
    ``normalize_text(normalize_text(x)) == normalize_text(x)``.
    """
    if text == "":
        return ""

    # 1) Drop embedded NULs that break JSON/DB text pipelines.
    cleaned = text.replace("\x00", "")

    # 2) CRLF first, then any remaining bare CR → LF.
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # 3) NBSP → ordinary space (keeps layout without special space code points).
    cleaned = cleaned.replace("\u00a0", " ")

    # 4) Trailing whitespace only; preserve leading spaces/tabs (indentation).
    lines = [line.rstrip(" \t") for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)

    # 5) At most two consecutive blank lines (three newline chars between text).
    cleaned = _MULTI_BLANK_LINES.sub("\n\n\n", cleaned)

    return cleaned
