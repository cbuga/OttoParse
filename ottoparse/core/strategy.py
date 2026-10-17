"""Swappable-step contracts.

Reusability is a core requirement: every stage of Phase 1 must be replaceable
without touching the pipeline. These protocols are the seams. Any object that
satisfies the protocol can be dropped in — no base class inheritance required
(structural typing), which keeps third-party implementations frictionless.

Two seams exist in Phase 1:

- :class:`ImageStep` — a single image-cleanup operation (deskew / denoise /
  binarize). Implementations transform a 2-D image array and may annotate the
  page record with flags and history.
- :class:`PageRouter` — decides, per PDF page, whether to take the text-layer
  path or the image path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from ottoparse.model.record import PageRecord, SourcePath


@runtime_checkable
class ImageStep(Protocol):
    """One swappable image-cleanup operation.

    Implementations should:
    - treat ``image`` as read-only and return a new array (no in-place mutation);
    - call ``record.add_step(...)`` to log that they ran and with what params;
    - call ``record.add_flag(...)`` for any quality issues they detect;
    - never raise for recoverable problems — flag and return best-effort output.
    """

    #: Logical step name, e.g. ``"deskew"``. Used in processing history.
    name: str

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        """Transform ``image`` and return the result.

        Args:
            image: 2-D (grayscale) or 3-D (color) image array.
            record: The page record; may be annotated with flags/history.

        Returns:
            The transformed image array.
        """
        ...


@dataclass(slots=True)
class RouteDecision:
    """The outcome of routing a single page."""

    path: SourcePath
    reason: str


@runtime_checkable
class PageRouter(Protocol):
    """Decides which Phase 1 branch a PDF page should take."""

    def decide(self, page: object) -> RouteDecision:
        """Inspect a PDF page and return a :class:`RouteDecision`.

        Args:
            page: A PyMuPDF ``Page`` (typed loosely to avoid a hard import in
                the protocol; implementations narrow this).

        Returns:
            A :class:`RouteDecision` with the chosen path and a human-readable
            reason recorded for provenance.
        """
        ...
