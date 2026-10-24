"""Background normalization — flatten stains and uneven paper before thresholding.

A stain, foxing, or uneven lighting darkens the paper *multiplicatively*: the
ink and the paper under a blot are both dimmed by roughly the same factor. A
local thresholder (Sauvola) sees the boundary between clean and stained paper as
a sharp brightness step and marks it as ink, so stain outlines come out as thick
black contours cutting through the text. CLAHE makes this worse by amplifying
those very edges.

:class:`BackgroundNormalizer` undoes the multiplicative darkening instead of
stretching it. It estimates the paper background with a grayscale morphological
closing (a disk larger than any text stroke fills the dark ink in with the
surrounding paper tone), smooths that estimate, and divides the page by it.
Clean paper and stained paper both map to near-white, while ink keeps its
contrast relative to its *own* local background — including ink inside a stain.
After this step even a global threshold works well.

Color can help: aged paper and most stains are yellow/brown, while printing ink
is dark in every channel. Measured on a real stained periodical scan, the
stain/paper ratio is 0.73 in luminance but 0.82 in the blue channel, while ink
stays near 0.5 in both — in blue, stains look more like paper. ``channel``
selects which channel is normalized; grayscale input always uses its single
channel.

The step outputs a 2-D ``uint8`` grayscale and is non-destructive in spirit: it
removes no strokes, it only re-expresses each pixel relative to its background.
"""

from __future__ import annotations

import cv2
import numpy as np

from ottoparse.imaging.utils import to_grayscale
from ottoparse.model.record import PageRecord

# Supported channel selections. Input arrays are RGB(A) (see ``ottoparse.io``).
_CHANNELS = {"luminance": None, "red": 0, "green": 1, "blue": 2}

# Auto kernel size as a fraction of the page's shorter side. ~0.02 gives ~41 px
# on a 300 DPI periodical page (~2000 px wide) — comfortably wider than a
# Matbu text stroke, yet small enough to follow a stain's edge.
_AUTO_KERNEL_FRACTION = 0.02
_MIN_KERNEL = 15


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


class BackgroundNormalizer:
    """Flat-field the page by dividing it by an estimated paper background.

    Args:
        channel: Which channel to normalize: ``"luminance"`` (standard
            grayscale), ``"red"``, ``"green"``, or ``"blue"``. ``"blue"``
            suppresses yellow/brown stains best on color scans. Grayscale input
            ignores this and uses its only channel (recorded in the history).
        kernel_size: Odd diameter (pixels) of the closing disk used to estimate
            the background. Must exceed the widest text stroke. ``None`` picks
            it automatically from the page size (~2% of the shorter side).
        blur_sigma: Gaussian sigma (pixels) applied to the background estimate
            so its blocky closing artifacts do not imprint on the output.
            ``None`` uses ``kernel_size / 5``.
    """

    name = "normalize_background"

    def __init__(
        self,
        channel: str = "luminance",
        kernel_size: int | None = None,
        blur_sigma: float | None = None,
    ) -> None:
        if channel not in _CHANNELS:
            raise ValueError(f"channel must be one of {sorted(_CHANNELS)}")
        if kernel_size is not None and (kernel_size < 3 or kernel_size % 2 == 0):
            raise ValueError("kernel_size must be an odd integer >= 3")
        if blur_sigma is not None and blur_sigma <= 0:
            raise ValueError("blur_sigma must be > 0")
        self.channel = channel
        self.kernel_size = kernel_size
        self.blur_sigma = blur_sigma

    def _select_channel(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        """Return the 2-D channel to normalize and the name actually used."""
        index = _CHANNELS[self.channel]
        if image.ndim == 2 or index is None:
            used = self.channel if image.ndim == 3 else "gray"
            return to_grayscale(image), used
        return np.ascontiguousarray(image[:, :, index]).astype(np.uint8), self.channel

    def _resolve_kernel(self, shape: tuple[int, ...]) -> int:
        if self.kernel_size is not None:
            return self.kernel_size
        return _odd(max(_MIN_KERNEL, round(min(shape[:2]) * _AUTO_KERNEL_FRACTION)))

    def estimate_background(self, gray: np.ndarray, kernel: int) -> np.ndarray:
        """Estimate the paper tone under every pixel (float32, >= 1)."""
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        # Closing = dilate then erode: dark ink narrower than the disk is filled
        # with the surrounding (brighter) paper tone; broad stains are kept.
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, disk)
        sigma = self.blur_sigma if self.blur_sigma is not None else kernel / 5
        background = cv2.GaussianBlur(background.astype(np.float32), (0, 0), sigma)
        return np.maximum(background, 1.0)

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        gray, used = self._select_channel(image)
        kernel = self._resolve_kernel(gray.shape)
        background = self.estimate_background(gray, kernel)
        flat = gray.astype(np.float32) / background * 255.0
        normalized = np.clip(flat, 0, 255).astype(np.uint8)
        record.add_step(
            self.name, "divide_by_closing",
            channel=used, kernel_size=kernel,
            blur_sigma=round(self.blur_sigma if self.blur_sigma is not None else kernel / 5, 2),
        )
        return normalized
