"""Pipeline configuration.

:class:`PipelineConfig` bundles the swappable components and the handful of
global knobs the Phase 1 pipeline needs. It deliberately holds *instances*
(already-constructed strategies) rather than classes, so callers configure each
component's parameters at construction time and the pipeline just runs them.

The default factories are defined where the concrete strategies live and wired
in by the top-level :class:`~ottoparse.pipeline.Pipeline`, so this module stays
free of heavy imports (OpenCV / scikit-image) until something is actually built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ottoparse.core.strategy import ImageStep, PageRouter


@dataclass(slots=True)
class PipelineConfig:
    """Everything needed to run Phase 1 on an input.

    Attributes:
        router: Strategy that decides text-layer vs image path for PDF pages.
        image_steps: Ordered list of image-cleanup steps (deskew, denoise,
            binarize). Runs in order; reorder or swap freely.
        rasterize_dpi: DPI used when rasterizing image-only PDF pages. 300 is a
            sensible default for OCR-grade historical print.
        write_images_to: If set, cleaned images are written under this directory
            and ``PageRecord.image_path`` is populated. If ``None``, images are
            kept in memory only.
    """

    router: PageRouter
    image_steps: list[ImageStep] = field(default_factory=list)
    rasterize_dpi: int = 300
    write_images_to: Path | None = None

    def __post_init__(self) -> None:
        if self.rasterize_dpi <= 0:
            raise ValueError("rasterize_dpi must be a positive integer")
        if self.write_images_to is not None:
            self.write_images_to = Path(self.write_images_to)
