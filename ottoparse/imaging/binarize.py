"""Adaptive binarization (diagram step 1.4).

The final image step converts a cleaned grayscale page to black-and-white.
Three interchangeable strategies are provided; all share the same interface and
all flag ``BINARIZATION_DEGENERATE`` if the result is nearly a single color
(a sign the input or parameters were wrong).

- :class:`OtsuBinarizer` (default) — global thresholding. Fast; works well
  once the background is flat, which is why the default chain runs
  ``BackgroundNormalizer`` first. Poor on raw uneven backgrounds. By default
  it also drops faint stand-alone marks with no dark core (foxing).
- :class:`SauvolaBinarizer` — local adaptive thresholding for pages that were
  not background-normalized. Its small window can mistake stain edges for ink.
- :class:`AdaptiveGaussianBinarizer` — OpenCV mean/Gaussian adaptive threshold.
  A lighter-weight local method, a middle ground between the other two.

Convention: output is ``uint8`` with foreground (text) = 0 (black) and
background = 255 (white).
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from ottoparse.imaging.utils import to_grayscale
from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord

# A page whose foreground fraction falls outside this band after binarization
# is almost certainly degenerate (all black or all white).
_MIN_FG = 0.0005
_MAX_FG = 0.9


def _check_degenerate(binary: np.ndarray, record: PageRecord, step: str) -> None:
    """Flag the record if the binary result is nearly a single color."""
    fg_fraction = float((binary == 0).mean())
    if fg_fraction < _MIN_FG or fg_fraction > _MAX_FG:
        record.add_flag(
            FlagCode.BINARIZATION_DEGENERATE, Severity.WARNING,
            measure=round(fg_fraction, 4),
            detail="binary output nearly single-color", step=step,
        )


class SauvolaBinarizer:
    """Local adaptive (Sauvola) binarization for non-normalized pages.

    Args:
        window_size: Odd side length of the local neighbourhood (pixels). Larger
            windows suit larger text / lower resolution.
        k: Sauvola sensitivity. Lower keeps more faint strokes; higher is
            stricter. 0.2 is the classic default.
    """

    name = "binarize"

    def __init__(self, window_size: int = 25, k: float = 0.2) -> None:
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        self.window_size = window_size
        self.k = k

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        gray = to_grayscale(image)
        thresh = threshold_sauvola(gray, window_size=self.window_size, k=self.k)
        # text (dark, below threshold) -> 0; background -> 255
        binary = np.where(gray > thresh, 255, 0).astype(np.uint8)
        _check_degenerate(binary, record, self.name)
        record.add_step(self.name, "sauvola", window_size=self.window_size, k=self.k)
        return binary


class OtsuBinarizer:
    """Global (Otsu) binarization — the default, after background normalization.

    On a flattened page Otsu's threshold sits high (paper is near-white, faded
    ink only mid-gray), so faint foxing spots and stain-edge mottling also fall
    below it and come out as solid black dots — next to a letter they read as a
    nokta. Printed ink, even faded, has a dark core; foxing does not. With
    ``core_contrast`` set, Otsu still defines every component's shape, but a
    component is kept only if its darkest pixel reaches a stricter "core"
    level (hysteresis thresholding):

        core = ink_level + core_contrast * (otsu - ink_level)

    where ``ink_level`` is the 10th percentile of the pixels Otsu marked as ink.
    Thin upright strokes (an alif, a rule) are blurred lighter than their
    width suggests, so components at least ``upright_aspect`` times taller
    than wide are always kept.

    Marks *touching* a glyph merge into its component and are kept; this step
    only removes faint components that stand alone.

    Args:
        core_contrast: Position of the core level between the ink level (0)
            and Otsu's threshold (1). Lower keeps more faint marks. ``None``
            disables core filtering (plain Otsu).
        upright_aspect: Height/width ratio at or above which a faint component
            is kept regardless of its core.
    """

    name = "binarize"

    # Percentile of the ink pixels taken as the page's typical ink darkness.
    _INK_PERCENTILE = 10

    def __init__(self, core_contrast: float | None = 0.4, upright_aspect: float = 2.5) -> None:
        if core_contrast is not None and not 0.0 < core_contrast < 1.0:
            raise ValueError("core_contrast must be in (0, 1) or None")
        if upright_aspect <= 0:
            raise ValueError("upright_aspect must be > 0")
        self.core_contrast = core_contrast
        self.upright_aspect = upright_aspect

    def _drop_coreless(self, gray: np.ndarray, binary: np.ndarray, otsu: float) -> int:
        """Clear faint, non-upright components in place; return how many."""
        ink = binary == 0
        if not ink.any():
            return 0
        ink_level = float(np.percentile(gray[ink], self._INK_PERCENTILE))
        core = ink_level + self.core_contrast * (otsu - ink_level)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            ink.astype(np.uint8), connectivity=8
        )
        # Darkest pixel per component: scatter-min of gray over the labels.
        darkest = np.full(n, 255, np.uint8)
        np.minimum.at(darkest, labels[ink], gray[ink])
        upright = stats[:, cv2.CC_STAT_HEIGHT] >= self.upright_aspect * stats[:, cv2.CC_STAT_WIDTH]
        drop = (darkest >= core) & ~upright
        drop[0] = False  # background label
        binary[drop[labels]] = 255
        return int(drop.sum())

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        gray = to_grayscale(image)
        otsu, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.core_contrast is None:
            _check_degenerate(binary, record, self.name)
            record.add_step(self.name, "otsu")
            return binary
        removed = self._drop_coreless(gray, binary, otsu)
        _check_degenerate(binary, record, self.name)
        record.add_step(
            self.name, "otsu_hysteresis",
            core_contrast=self.core_contrast, upright_aspect=self.upright_aspect,
            removed_faint=removed,
        )
        return binary


class AdaptiveGaussianBinarizer:
    """OpenCV adaptive-Gaussian thresholding — a lighter local method.

    Args:
        block_size: Odd neighbourhood size for the local threshold.
        c: Constant subtracted from the local weighted mean.
    """

    name = "binarize"

    def __init__(self, block_size: int = 31, c: float = 10.0) -> None:
        if block_size % 2 == 0:
            raise ValueError("block_size must be odd")
        self.block_size = block_size
        self.c = c

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        gray = to_grayscale(image)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, self.block_size, self.c,
        )
        _check_degenerate(binary, record, self.name)
        record.add_step(
            self.name, "adaptive_gaussian", block_size=self.block_size, c=self.c
        )
        return binary
