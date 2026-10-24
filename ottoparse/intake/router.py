"""Per-page routing: text-layer path vs image path (diagram step 1.1).

A real archival PDF is often *mixed*: some pages carry a clean text layer, some
are scanned images with no text, and some have a broken embedded OCR layer
producing mojibake. So the decision is made **per page**, not per document.

:class:`DefaultRouter` applies a transparent heuristic:

1. Extract the page's text. If there is very little of it, the page is almost
   certainly a scan -> image path.
2. If there is enough text but it is not plausibly Perso-Arabic (low Arabic
   ratio), treat the text layer as suspect -> image path (re-OCR downstream).
3. Otherwise the text layer is usable -> text path.

Every decision carries a human-readable ``reason`` so routing stays auditable,
and the router is swappable so "clean enough" can be tuned on a real corpus.
"""

from __future__ import annotations

from ottoparse.core.strategy import PageRouter, RouteDecision
from ottoparse.intake.pdf_text import extract_page_text
from ottoparse.intake.script import arabic_ratio
from ottoparse.model.record import SourcePath


class DefaultRouter(PageRouter):
    """Heuristic text-vs-image router for PDF pages.

    Args:
        min_chars: Minimum number of non-whitespace characters for a page's
            text layer to be considered at all. Below this, route to image.
        min_arabic_ratio: Minimum fraction of letters that must be Perso-Arabic
            for the text layer to be trusted. Below this (but above
            ``min_chars``), the layer is flagged suspect and routed to image.
            Set to ``0.0`` to accept any script (e.g. Latin-only documents).
    """

    def __init__(self, min_chars: int = 50, min_arabic_ratio: float = 0.30) -> None:
        if min_chars < 0:
            raise ValueError("min_chars must be non-negative")
        if not 0.0 <= min_arabic_ratio <= 1.0:
            raise ValueError("min_arabic_ratio must be between 0.0 and 1.0")
        self.min_chars = min_chars
        self.min_arabic_ratio = min_arabic_ratio

    def decide(self, page: object) -> RouteDecision:
        text = extract_page_text(page)
        char_count = len(text.strip().replace(" ", "").replace("\n", ""))

        if char_count < self.min_chars:
            return RouteDecision(
                path=SourcePath.IMAGE,
                reason=f"thin/absent text layer: {char_count} chars "
                f"(< {self.min_chars}); routing to image",
            )

        ratio = arabic_ratio(text)
        if ratio < self.min_arabic_ratio:
            return RouteDecision(
                path=SourcePath.IMAGE,
                reason=f"suspect text layer: {char_count} chars but "
                f"{ratio:.2f} arabic-ratio (< {self.min_arabic_ratio}); "
                f"re-OCR downstream",
            )

        return RouteDecision(
            path=SourcePath.TEXT_LAYER,
            reason=f"usable text layer: {char_count} chars, "
            f"{ratio:.2f} arabic-ratio",
        )
