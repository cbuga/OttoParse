"""Contrast enhancement — CLAHE for degraded historical pages.

On yellowed, stained paper the text-to-background contrast varies wildly across
the page: crisp in clean areas, almost nothing where a stain or foxing darkens
the paper. A global contrast stretch cannot fix this because it applies one
mapping everywhere. CLAHE (Contrast Limited Adaptive Histogram Equalization)
equalizes contrast in local tiles, so each region — including a stained patch —
gets its own stretch, pulling faint ink apart from its local background.

:class:`ClaheEnhancer` wraps OpenCV's CLAHE. It runs on grayscale (converting
internally if handed color), **before** denoising and binarization. It is
opt-in: CLAHE also amplifies speckle, foxing texture, and bleed-through, so on
already-clean pages it can hurt more than help. The ``clip_limit`` caps local
amplification and is the main guard against over-enhancement.
"""

from __future__ import annotations

import cv2
import numpy as np

from ottoparse.imaging.utils import to_grayscale
from ottoparse.model.record import PageRecord


class ClaheEnhancer:
    """Local contrast enhancement via OpenCV CLAHE.

    Args:
        clip_limit: Contrast-amplification cap. Higher values enhance more
            aggressively and amplify more noise; 2.0-3.0 is a safe range for
            historical print. Must be > 0.
        tile_grid_size: Number of tiles per axis for the local equalization
            ``(cols, rows)``. Smaller grids act more globally; larger grids adapt
            to finer local variation. 8x8 is a sensible default.
    """

    name = "enhance_contrast"

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        if clip_limit <= 0:
            raise ValueError("clip_limit must be > 0")
        gx, gy = tile_grid_size
        if gx < 1 or gy < 1:
            raise ValueError("tile_grid_size entries must be >= 1")
        self.clip_limit = clip_limit
        self.tile_grid_size = (int(gx), int(gy))

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        gray = to_grayscale(image)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        enhanced = clahe.apply(gray)
        record.add_step(
            self.name, "clahe",
            clip_limit=self.clip_limit, tile_grid_size=list(self.tile_grid_size),
        )
        return enhanced
