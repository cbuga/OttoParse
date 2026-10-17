"""Quality flags attached to page records.

Phase 1 never silently drops a page. When a step detects a problem it attaches
an :class:`IssueFlag` to the record and continues on a best-effort basis. The
downstream consumer (Phase 2, or the user) decides what to do with flags based
on their :class:`Severity`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum


class Severity(Enum):
    """How serious an :class:`IssueFlag` is.

    - ``INFO``    — informational; the step succeeded but is worth noting.
    - ``WARNING`` — a quality concern; output is usable but may be degraded.
    - ``ERROR``   — the step could not complete; output may be missing or raw.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FlagCode(StrEnum):
    """Stable, machine-readable codes for known Phase 1 issues.

    Using an enum (rather than free-form strings) keeps the vocabulary
    consistent so downstream filtering can rely on exact codes. As a
    :class:`~enum.StrEnum`, members are plain strings so flags serialize
    cleanly to JSON.
    """

    # --- Routing (1.1) ---
    NO_TEXT_LAYER = "NO_TEXT_LAYER"          # PDF page had no extractable text
    SUSPECT_TEXT_LAYER = "SUSPECT_TEXT_LAYER"  # text present but looks like mojibake
    THIN_TEXT_LAYER = "THIN_TEXT_LAYER"      # very little text; routed to image path

    # --- Deskew (1.2) ---
    SEVERE_SKEW = "SEVERE_SKEW"              # estimated skew beyond safe range
    SKEW_UNDETECTED = "SKEW_UNDETECTED"      # could not estimate a skew angle

    # --- Denoise (1.3) ---
    HIGH_NOISE = "HIGH_NOISE"                # noise estimate above threshold
    BLEED_THROUGH = "BLEED_THROUGH"          # verso ink showing through the page
    LOW_CONTRAST = "LOW_CONTRAST"            # faded / low dynamic range

    # --- Binarization (1.4) ---
    BINARIZATION_DEGENERATE = "BINARIZATION_DEGENERATE"  # output nearly all one color

    # --- General / IO ---
    UNREADABLE_PAGE = "UNREADABLE_PAGE"      # page could not be decoded at all
    PROCESSING_ERROR = "PROCESSING_ERROR"    # an unexpected error during a step


@dataclass(slots=True)
class IssueFlag:
    """A single quality annotation on a :class:`~ottoparse.model.record.PageRecord`.

    Attributes:
        code: A :class:`FlagCode` identifying the issue.
        severity: How serious it is.
        measure: Optional numeric measurement that triggered the flag
            (e.g. estimated skew angle in degrees, a noise estimate). Lets the
            downstream "confidence filtering" stage threshold on real values.
        detail: Optional human-readable note for debugging.
        step: Name of the step that raised the flag (e.g. ``"deskew"``).
    """

    code: FlagCode
    severity: Severity
    measure: float | None = None
    detail: str | None = None
    step: str | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "measure": self.measure,
            "detail": self.detail,
            "step": self.step,
        }
