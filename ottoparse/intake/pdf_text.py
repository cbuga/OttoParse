"""Deterministic PDF text extraction and normalization.

When a PDF page already carries a usable text layer we extract it directly
(PyMuPDF) and apply light, deterministic regex normalization. This is the
"bypass OCR" path from the plan: no recognition, no model, just clean-up of
text that already exists.

Normalization is intentionally conservative. We fix whitespace, strip control
characters, repair words split across line breaks, and apply Unicode NFC so the
output is canonical — but we do *not* transliterate or rewrite orthography. That
belongs to Phase 3, which must receive the original script intact.
"""

from __future__ import annotations

import re
import unicodedata

# Control characters except the whitespace we normalize separately.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Any run of horizontal whitespace (spaces, tabs) -> a single space.
_HSPACE_RE = re.compile(r"[ \t\u00a0]+")
# Three or more newlines -> a paragraph break (two newlines).
_MULTI_NL_RE = re.compile(r"\n{3,}")
# A letter, a hyphen, then a newline: a word hyphenated across a line break.
# Latin only — Perso-Arabic print does not hyphenate words this way.
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])-\n(?=[A-Za-z])")
# Trailing spaces before a newline.
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")


def normalize_text(text: str) -> str:
    """Apply conservative, deterministic normalization to extracted text.

    Steps, in order:
    1. Unicode NFC normalization (canonical composition).
    2. Remove control characters.
    3. Rejoin Latin words hyphenated across a line break.
    4. Collapse horizontal whitespace runs to a single space.
    5. Trim trailing whitespace on each line.
    6. Collapse 3+ blank lines to a single paragraph break.
    7. Strip leading/trailing whitespace of the whole block.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1", text)
    text = _HSPACE_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def extract_page_text(page: object) -> str:
    """Extract raw text from a PyMuPDF page.

    Uses the default text extraction. ``page`` is typed loosely to keep this
    module importable without PyMuPDF types leaking into signatures; at runtime
    it must be a ``fitz.Page``.
    """
    # PyMuPDF pages expose get_text(); "text" preserves reading order reasonably.
    return page.get_text("text")  # type: ignore[attr-defined]
