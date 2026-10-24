"""Script-detection helpers used by routing.

Late-Ottoman printed text is in the Perso-Arabic script. When a PDF page has an
embedded text layer we want to know whether that text is *plausibly* Ottoman
(worth keeping) or garbage/mojibake from a broken embedded OCR layer (in which
case we would rather rasterize and re-OCR downstream).

The heuristic here is deliberately simple and transparent: count how many
letters fall in the Arabic Unicode ranges versus how many letters there are in
total. It is a signal, not a verdict; the router thresholds on it and records
the ratio in provenance so decisions stay auditable.
"""

from __future__ import annotations

# Unicode blocks that carry Perso-Arabic script used in Ottoman printing.
# - Arabic                      U+0600..U+06FF
# - Arabic Supplement           U+0750..U+077F
# - Arabic Extended-A           U+08A0..U+08FF
# - Arabic Presentation Forms-A U+FB50..U+FDFF
# - Arabic Presentation Forms-B U+FE70..U+FEFF
_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def is_arabic_char(ch: str) -> bool:
    """True if ``ch`` lies in one of the Perso-Arabic Unicode ranges."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_RANGES)


def arabic_ratio(text: str) -> float:
    """Fraction of *letters* in ``text`` that are Perso-Arabic.

    Punctuation, digits, and whitespace are ignored so that page numbers,
    spacing, and stray marks do not dilute the signal. Returns ``0.0`` for text
    with no letters at all.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for ch in letters if is_arabic_char(ch))
    return arabic / len(letters)
