"""Confidence mapping — flag faint/stained regions without deleting pixels.

The guardrail: faded characters that look like "noise" to a rigid cleaner are
often real Ottoman words or Persian izafet links. Instead of hard-deleting
ambiguous strokes, :class:`ConfidenceMap` *identifies* the low-confidence areas
and records them on the page (as :class:`~ottoparse.model.record.Region`s with a
confidence score) so a downstream lexicon matcher or language model can
reconstruct them from the preserved grayscale.

This step is **non-destructive**: it reads the grayscale and returns it
unchanged. It only annotates the record. It is best placed near the end of the
image chain, after cleaning but so it sees the grayscale tones (it converts
internally if handed a binary image, though it is most informative on grayscale).

How it scores a tile: a tile that contains ink the engine can read has good
local contrast (dark strokes against light paper). A faded or stained tile has
ink present but *low* contrast — strokes only slightly darker than their
background. So each tile's confidence is its local contrast (robust dynamic
range) normalized to 0-1; tiles that are confidently blank (no ink) are skipped,
and tiles that are low-contrast-but-not-blank are flagged.
"""

from __future__ import annotations

import numpy as np

from ottoparse.imaging.utils import to_grayscale
from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord

# Contrast span (0-255) that maps to confidence 1.0; tiles at/above this are
# considered clearly legible. Chosen so ordinary dark-on-light print scores high.
_FULL_CONTRAST = 120.0


class ConfidenceMap:
    """Flag faint/stained regions as low-confidence without altering pixels.

    The page is divided into a grid of tiles. For each tile the local contrast
    (95th minus 5th percentile of intensity) is measured. A tile is flagged as a
    low-confidence region when it:

    - contains ink (it is not confidently blank), yet
    - has contrast below ``min_contrast`` (ink present but faint/stained).

    Adjacent flagged tiles are merged into larger regions so a stain becomes one
    bounding box rather than many. Nothing is deleted.

    The threshold is **adaptive by default**: a fixed absolute contrast value
    cannot tell a crisp page from a uniformly-faded one. Instead the cutoff is a
    low percentile of *this page's own* non-blank tile contrasts, so only tiles
    that are faint relative to the page's typical text are flagged. An absolute
    override is available when you want a fixed cutoff.

    Args:
        tile: Side length (pixels) of each analysis tile.
        low_percentile: Tiles whose contrast falls in this bottom percentile of
            the page's non-blank tiles are flagged (adaptive mode).
        min_contrast: Optional absolute contrast cutoff (0-255). If set, it
            overrides ``low_percentile`` and flags any non-blank tile below it.
        blank_ink_fraction: A tile with fewer than this fraction of "ink-ish"
            (below-mean) pixels is treated as blank paper and skipped.
    """

    name = "confidence_map"

    def __init__(
        self,
        tile: int = 64,
        low_percentile: float = 20.0,
        min_contrast: float | None = None,
        blank_ink_fraction: float = 0.02,
    ) -> None:
        if tile < 8:
            raise ValueError("tile must be >= 8")
        if not 0.0 < low_percentile < 100.0:
            raise ValueError("low_percentile must be in (0, 100)")
        self.tile = tile
        self.low_percentile = low_percentile
        self.min_contrast = min_contrast
        self.blank_ink_fraction = blank_ink_fraction

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        gray = to_grayscale(image)
        h, w = gray.shape
        t = self.tile
        ny, nx = (h + t - 1) // t, (w + t - 1) // t

        # Pass 1: measure contrast of every non-blank tile.
        contrast_grid = np.full((ny, nx), np.nan, dtype=np.float32)
        conf = np.ones((ny, nx), dtype=np.float32)
        page_mean = float(gray.mean())
        for j in range(ny):
            for i in range(nx):
                tile = gray[j * t:(j + 1) * t, i * t:(i + 1) * t]
                if tile.size == 0:
                    continue
                if float((tile < page_mean).mean()) < self.blank_ink_fraction:
                    continue  # confidently blank paper
                lo, hi = np.percentile(tile, [5, 95])
                contrast = float(hi - lo)
                contrast_grid[j, i] = contrast
                conf[j, i] = min(contrast / _FULL_CONTRAST, 1.0)

        non_blank = contrast_grid[~np.isnan(contrast_grid)]
        if non_blank.size == 0:
            record.add_step(self.name, "tile_contrast", tile=self.tile, regions=0)
            return image

        # Decide the cutoff: absolute override, else adaptive (page percentile).
        if self.min_contrast is not None:
            cutoff = self.min_contrast
            mode = f"absolute<{self.min_contrast}"
        else:
            cutoff = float(np.percentile(non_blank, self.low_percentile))
            mode = f"p{self.low_percentile:g}={cutoff:.1f}"

        # Pass 2: flag non-blank tiles below the cutoff.
        low = (~np.isnan(contrast_grid)) & (contrast_grid < cutoff)

        regions = _merge_tiles(low, t, w, h)
        for (x, y, rw, rh) in regions:
            # region confidence = mean confidence of its tiles
            ti0, tj0 = x // t, y // t
            ti1, tj1 = (x + rw + t - 1) // t, (y + rh + t - 1) // t
            region_conf = float(conf[tj0:tj1, ti0:ti1].mean())
            record.add_region(
                x, y, rw, rh, kind="low_confidence",
                confidence=round(region_conf, 3),
                detail="faint/low-contrast ink (possible faded or stained text); "
                "preserved for downstream reconstruction",
            )

        if regions:
            # a page-level flag summarizing the fraction of low-confidence area
            low_area = sum(rw * rh for (_, _, rw, rh) in regions)
            record.add_flag(
                FlagCode.LOW_CONFIDENCE_REGION, Severity.WARNING,
                measure=round(low_area / float(w * h), 4),
                detail=f"{len(regions)} low-confidence region(s) flagged for "
                f"downstream reconstruction (not deleted)",
                step=self.name,
            )
        record.add_step(
            self.name, "tile_contrast",
            tile=self.tile, cutoff=mode, regions=len(regions),
        )
        # Non-destructive: return the input untouched.
        return image


def _merge_tiles(
    low: np.ndarray, t: int, w: int, h: int
) -> list[tuple[int, int, int, int]]:
    """Merge a boolean tile grid into pixel bounding boxes via connected labels.

    Uses a simple flood over 4-connected True tiles so that a contiguous stain
    becomes one box. Returns (x, y, width, height) clamped to page bounds.
    """
    ny, nx = low.shape
    seen = np.zeros_like(low)
    boxes: list[tuple[int, int, int, int]] = []

    for sj in range(ny):
        for si in range(nx):
            if not low[sj, si] or seen[sj, si]:
                continue
            # BFS over connected low tiles
            stack = [(sj, si)]
            seen[sj, si] = True
            min_i = max_i = si
            min_j = max_j = sj
            while stack:
                cj, ci = stack.pop()
                min_i, max_i = min(min_i, ci), max(max_i, ci)
                min_j, max_j = min(min_j, cj), max(max_j, cj)
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nj, ni = cj + dj, ci + di
                    if 0 <= nj < ny and 0 <= ni < nx and low[nj, ni] and not seen[nj, ni]:
                        seen[nj, ni] = True
                        stack.append((nj, ni))
            x, y = min_i * t, min_j * t
            rw = min((max_i + 1) * t, w) - x
            rh = min((max_j + 1) * t, h) - y
            boxes.append((x, y, rw, rh))
    return boxes
