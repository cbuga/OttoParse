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
class PageRecord:
    """One page of Phase 1 output.

    Exactly one of ``text`` / ``image`` is populated, matching ``source_path``.
    ``image_path`` is set only when the cleaned image was written to disk.
    """

    provenance: Provenance
    text: str | None = None
    image: np.ndarray | None = None
    image_path: Path | None = None
    flags: list[IssueFlag] = field(default_factory=list)
    history: list[ProcessingStep] = field(default_factory=list)

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
            "flags": [f.to_dict() for f in self.flags],
            "history": [s.to_dict() for s in self.history],
        }
        if include_image and self.image is not None:
            d["image_shape"] = list(self.image.shape)
        return d
