"""
Conservative operational-safety policy for the experimental assistant.

This module does NOT claim comprehensive prompt-injection resistance or
factual entailment. It applies an explicitly scoped heuristic: the prototype
must not return unverified physical-equipment repair, replacement, electrical
reset, or safety-bypass instructions.

Limitations (document clearly):
- Keyword / pattern guards have false positives (benign text mentioning
  "fuse" or "breaker") and false negatives (paraphrased unsafe advice).
- A valid evidence_id citation is not proof the answer is entailed; there is
  no factual entailment verifier.
- Prompt changes alone do not eliminate injection; this guard is a last-line
  abstention layer for a small, known class of unsafe outputs.
- Passing the current fixture set does not imply resistance to unseen attacks;
  future evaluation against novel paraphrases and injection variants is required.
- Operational-safety abstention is distinct from insufficient-evidence abstention.
"""

from __future__ import annotations

import re

OPERATIONAL_SAFETY_ABSTAIN_ANSWER = (
    "This experimental assistant cannot provide unverified operational "
    "repair, replacement, electrical reset, or safety-bypass instructions. "
    "Consult qualified personnel and the manufacturer documentation."
)

# Question intents that request actionable field work rather than description.
_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcorrective\s+action\b", re.IGNORECASE),
    re.compile(r"\breplace(?:ment)?\s+(?:the\s+)?fuse\b", re.IGNORECASE),
    re.compile(r"\breset(?:\s+the)?\s+(?:circuit\s+)?breaker\b", re.IGNORECASE),
    re.compile(r"\bexact\s+replacement\s+part\s+number\b", re.IGNORECASE),
    re.compile(r"\bsafety\s+bypass\b", re.IGNORECASE),
    re.compile(r"\bdefeat\s+(?:the\s+)?interlock\b", re.IGNORECASE),
    re.compile(r"\bjumper\s+out\b", re.IGNORECASE),
    re.compile(
        r"\b(?:what\s+should\s+i\s+do|how\s+do\s+i\s+fix)\b.*\bblower\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bblower\b.*\b(?:fails?|failed|not\s+operat)\b",
        re.IGNORECASE,
    ),
)

# Answer / evidence content that issues or echoes unsafe operational advice.
_INSTRUCTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\breplace(?:\s+the)?\s+fuse\b", re.IGNORECASE),
    re.compile(r"\breset(?:\s+the)?\s+(?:circuit\s+)?breaker\b", re.IGNORECASE),
    re.compile(r"\bsafety\s+bypass\b", re.IGNORECASE),
    re.compile(r"\bdefeat\s+(?:the\s+)?interlock\b", re.IGNORECASE),
    re.compile(r"\bjumper\s+out\b", re.IGNORECASE),
    re.compile(
        r"\binitiate\s+a\s+blower\s+shutdown\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcorrective\s+action\s+is\s+replace\b",
        re.IGNORECASE,
    ),
)


def requests_unverified_operational_instruction(question: str) -> bool:
    """
    Return True when the user question asks for scoped unsafe operational help.

    Informational questions about sensors, filters, or equipment behavior that
    do not request repair/reset/replacement actions should return False.
    """
    text = (question or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _REQUEST_PATTERNS)


def contains_unverified_operational_instruction(text: str) -> bool:
    """
    Return True when text issues scoped unsafe operational instructions.

    Used on model answers (and optionally evidence) as a fail-closed guard.
    """
    body = (text or "").strip()
    if not body:
        return False
    return any(pattern.search(body) for pattern in _INSTRUCTION_PATTERNS)
