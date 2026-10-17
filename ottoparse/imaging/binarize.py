"""Adaptive binarization (diagram step 1.4).

The final image step converts a cleaned grayscale page to black-and-white.
Three interchangeable strategies are provided; all share the same interface and
all flag ``BINARIZATION_DEGENERATE`` if the result is nearly a single color
(a sign the input or parameters were wrong).

- :class:`SauvolaBinarizer` (default) — local adaptive thresholding. Best for
  degraded historical scans with uneven lighting, staining, and faded ink,
  because the threshold adapts to each neighbourhood.
- :class:`OtsuBinarizer` — global thresholding. Fast; fine for clean, evenly
  lit pages but poor on uneven backgrounds.
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
    """Local adaptive (Sauvola) binarization — the default for degraded scans.

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
    """Global (Otsu) binarization — fast, best on clean evenly-lit pages."""

    name = "binarize"

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _check_degenerate(binary, record, self.name)
        record.add_step(self.name, "otsu")
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
