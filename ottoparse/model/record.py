"""The Phase 1 -> Phase 2 hand-off contract.

Every input page becomes exactly one :class:`PageRecord`. A record carries:

- **provenance** — where it came from and which path it took, and why;
- **payload** — either extracted text (text-layer path) or a cleaned image
  (image path), never silently both;
- **diagnostics** — a list of :class:`~ottoparse.model.flags.IssueFlag`;
- **history** — which processing steps ran, with which strategy and parameters.

A downstream OCR/HTR stage inspects ``source_path`` to decide whether a page is
already text or still needs recognition, and can filter on ``flags``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ottoparse.model.flags import FlagCode, IssueFlag, Severity

if TYPE_CHECKING:
    import numpy as np


class SourcePath(Enum):
    """Which Phase 1 branch produced a page's payload."""

    TEXT_LAYER = "text_layer"  # extracted from a PDF text layer; already clean
    IMAGE = "image"            # rasterized/scanned; cleaned via the image pipeline


@dataclass(slots=True)
class Provenance:
    """Where a page came from and how it was routed.

    Attributes:
        source_file: Path to the original input file.
        page_index: 0-based page index within the source document.
        source_path: Which branch the page took (:class:`SourcePath`).
        path_reason: Human-readable justification for the routing decision,
            e.g. ``"text layer: 1240 chars, 0.92 arabic-range ratio"``. Kept as
            free text because it is for auditing, not machine branching.
    """

    source_file: Path
    page_index: int
    source_path: SourcePath
    path_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "source_file": str(self.source_file),
            "page_index": self.page_index,
            "source_path": self.source_path.value,
            "path_reason": self.path_reason,
        }


@dataclass(slots=True)
class ProcessingStep:
    """A record of one step having run on a page.

    Attributes:
        name: Logical step name (``"deskew"``, ``"denoise"``, ``"binarize"``).
        strategy: The concrete implementation that ran (``"sauvola"``, ``"otsu"``).
        params: The parameters the strategy used, for reproducibility.
    """

    name: str
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "strategy": self.strategy, "params": self.params}


@dataclass(slots=True)
class Region:
    """A rectangular area of a page flagged for downstream attention.

    Used to mark low-confidence zones (faint ink, stains) without deleting the
    pixels, so a lexicon matcher or language model can reconstruct the content
    from the preserved grayscale.

    Coordinates are pixel bounds on the page: ``(x, y)`` top-left, plus width
    and height.

    Attributes:
        x, y, width, height: Bounding box in pixels.
        kind: What the region represents (e.g. ``"low_confidence"``, ``"stain"``).
        confidence: A 0-1 score; lower means more doubtful. For a stained patch
            this is roughly the local contrast, so downstream can rank regions.
        detail: Optional human-readable note.
    """

    x: int
    y: int
    width: int
    height: int
    kind: str = "low_confidence"
    confidence: float | None = None
    detail: str | None = None

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "kind": self.kind,
            "confidence": self.confidence,
            "detail": self.detail,
        }


@dataclass(slots=True)
class PageRecord:
    """One page of Phase 1 output.

    For a text-layer page, ``text`` is populated. For an image page:

    - ``gray`` holds the cleaned, **pre-binarization grayscale** — the lossless
      representation. Binarization is irreversible and drops every faint stroke
      that falls below threshold (exactly the text under a stain), so the
      grayscale is preserved so Phase 2/3 can recover ambiguous glyphs.
    - ``image`` holds the binary (black/white) result — a convenience view for
      engines that want it, derived from the grayscale.

    ``image_path`` / ``gray_path`` are set only when written to disk.
    ``regions`` holds low-confidence regions (e.g. stains) flagged for
    downstream reconstruction rather than deleted.
    """

    provenance: Provenance
    text: str | None = None
    image: np.ndarray | None = None
    gray: np.ndarray | None = None
    image_path: Path | None = None
    gray_path: Path | None = None
    flags: list[IssueFlag] = field(default_factory=list)
    history: list[ProcessingStep] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)

    # --- convenience helpers -------------------------------------------------

    def add_flag(
        self,
        code: FlagCode,
        severity: Severity,
        *,
        measure: float | None = None,
        detail: str | None = None,
        step: str | None = None,
    ) -> IssueFlag:
        """Attach an issue flag and return it."""
        flag = IssueFlag(
            code=code, severity=severity, measure=measure, detail=detail, step=step
        )
        self.flags.append(flag)
        return flag

    def add_step(self, name: str, strategy: str, **params: Any) -> ProcessingStep:
        """Record that a processing step ran, and return it."""
        step = ProcessingStep(name=name, strategy=strategy, params=params)
        self.history.append(step)
        return step

    def add_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        kind: str = "low_confidence",
        confidence: float | None = None,
        detail: str | None = None,
    ) -> Region:
        """Record a low-confidence region and return it."""
        region = Region(
            x=x, y=y, width=width, height=height,
            kind=kind, confidence=confidence, detail=detail,
        )
        self.regions.append(region)
        return region

    @property
    def is_text(self) -> bool:
        return self.provenance.source_path is SourcePath.TEXT_LAYER

    @property
    def is_image(self) -> bool:
        return self.provenance.source_path is SourcePath.IMAGE

    @property
    def has_errors(self) -> bool:
        """True if any attached flag is ERROR severity."""
        return any(f.severity is Severity.ERROR for f in self.flags)

    @property
    def max_severity(self) -> Severity | None:
        """The most serious severity among flags, or None if clean."""
        order = {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2}
        if not self.flags:
            return None
        return max((f.severity for f in self.flags), key=lambda s: order[s])

    def to_dict(self, *, include_image: bool = False) -> dict:
        """Serialize to a plain dict (JSON-friendly).

        The raw image array is omitted by default (it is large and not
        JSON-serializable); ``image_path`` is always included so the cleaned
        image can be located on disk.
        """
        d: dict[str, Any] = {
            "provenance": self.provenance.to_dict(),
            "text": self.text,
            "image_path": str(self.image_path) if self.image_path else None,
            "gray_path": str(self.gray_path) if self.gray_path else None,
            "flags": [f.to_dict() for f in self.flags],
            "history": [s.to_dict() for s in self.history],
            "regions": [r.to_dict() for r in self.regions],
        }
        if include_image and self.image is not None:
            d["image_shape"] = list(self.image.shape)
        return d
