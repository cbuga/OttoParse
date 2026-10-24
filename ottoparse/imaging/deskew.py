"""Deskewing & alignment (diagram step 1.2).

Printed pages are often scanned at a small rotation. Left uncorrected, skew
hurts line detection and recognition downstream. Both strategies here estimate
the dominant skew angle and rotate the page to level it; they differ only in
the estimator.

- :class:`ProjectionDeskew` (the default) searches for the rotation that makes
  the horizontal projection profile of the ink sharpest: when text lines are
  level, ink piles up in narrow bands with clean gaps between them. It works on
  a background-flattened, downscaled copy and keeps only glyph-sized connected
  components, so yellowed paper, stains, dark scan margins, and column rules do
  not steer the estimate. It needs no long straight segments, which is what
  Ottoman Turkish print in Perso-Arabic script lacks.
- :class:`HoughDeskew` takes the median angle of long near-horizontal segments
  found by the probabilistic Hough transform. It suits Latin print with
  continuous baselines, but on late-Ottoman periodicals it finds no segment
  long enough (word gaps break every line, and pages are set in narrow
  columns), so it flags ``SKEW_UNDETECTED`` on real scans.

If the estimated angle exceeds a safe range the page is still rotated
best-effort but flagged ``SEVERE_SKEW``; if no angle can be estimated the page
is left as-is and flagged ``SKEW_UNDETECTED``.
"""

from __future__ import annotations

import cv2
import numpy as np

from ottoparse.imaging.background import BackgroundNormalizer
from ottoparse.imaging.utils import to_grayscale
from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord


class _Deskew:
    """Shared apply/rotate logic; subclasses provide :meth:`estimate_angle`.

    Args:
        max_angle: Angles beyond +/- this (degrees) are treated as unreliable;
            the page is flagged ``SEVERE_SKEW`` and the correction is clamped.
        min_correction: Angles with magnitude below this (degrees) are ignored;
            the page is considered already level (avoids needless resampling).
    """

    name = "deskew"
    #: Estimator name recorded in the processing history.
    method = ""
    #: Flag detail used when :meth:`estimate_angle` returns None.
    undetected_detail = ""

    def __init__(self, max_angle: float = 15.0, min_correction: float = 0.1) -> None:
        self.max_angle = max_angle
        self.min_correction = min_correction

    def estimate_angle(self, gray: np.ndarray) -> float | None:
        """Estimate skew in degrees (positive = counter-clockwise), or None."""
        raise NotImplementedError

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # White border so rotated-in corners match page background.
        border = 255 if image.ndim == 2 else (255, 255, 255)
        return cv2.warpAffine(
            image, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        gray = to_grayscale(image)
        angle = self.estimate_angle(gray)

        if angle is None:
            record.add_flag(
                FlagCode.SKEW_UNDETECTED, Severity.WARNING,
                detail=self.undetected_detail, step=self.name,
            )
            record.add_step(self.name, self.method, angle=None, applied=False)
            return image

        if abs(angle) > self.max_angle:
            record.add_flag(
                FlagCode.SEVERE_SKEW, Severity.WARNING, measure=abs(angle),
                detail=f"estimated {angle:.2f} deg exceeds +/-{self.max_angle}",
                step=self.name,
            )
            angle = float(np.clip(angle, -self.max_angle, self.max_angle))

        if abs(angle) < self.min_correction:
            record.add_step(self.name, self.method, angle=angle, applied=False)
            return image

        rotated = self._rotate(image, angle)
        record.add_step(self.name, self.method, angle=round(angle, 3), applied=True)
        return rotated


class HoughDeskew(_Deskew):
    """Estimate and correct page skew via the probabilistic Hough transform.

    Args:
        max_angle: See :class:`ProjectionDeskew`.
        min_correction: See :class:`ProjectionDeskew`.
    """

    method = "hough"
    undetected_detail = "no near-horizontal lines found"

    def estimate_angle(self, gray: np.ndarray) -> float | None:
        """Estimate skew in degrees (positive = counter-clockwise), or None."""
        # Binarize a working copy first so edge detection keys on text/rules,
        # not scanner noise. Deskew runs before denoise in the pipeline, so the
        # incoming image may be noisy; Otsu + a light blur gives the Hough
        # transform clean, connected baselines to lock onto.
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Join characters along each line into bars so baselines dominate.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
        min_len = max(gray.shape[1] // 4, 50)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=min_len, maxLineGap=20,
        )
        if lines is None:
            return None

        # OpenCV <5 returns shape (N, 1, 4); OpenCV >=5 returns (N, 4). Reshape
        # to (N, 4) so iteration works across versions.
        segments = lines.reshape(-1, 4)

        angles: list[float] = []
        for x1, y1, x2, y2 in segments:
            angle = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
            # Keep near-horizontal segments only (text baselines / rules).
            if -45 < angle < 45:
                angles.append(angle)
        if not angles:
            return None
        return float(np.median(angles))


class ProjectionDeskew(_Deskew):
    """Estimate and correct page skew by maximizing projection-profile sharpness.

    Args:
        max_angle: Angles beyond +/- this (degrees) are treated as unreliable;
            the page is flagged ``SEVERE_SKEW`` and the correction is clamped.
            The search covers twice this range (capped at 45 degrees) so a
            severe skew is still measured and flagged.
        min_correction: Angles with magnitude below this (degrees) are ignored;
            the page is considered already level (avoids needless resampling).
        work_size: The longer side (pixels) of the downscaled working copy the
            angle is estimated on. ~1200 px keeps text lines several pixels
            apart on a full periodical page while making the search cheap.
        coarse_step: Search step (degrees) over ``+/- max_angle``.
        fine_step: Refinement step (degrees) around the coarse optimum.
    """

    method = "projection"
    undetected_detail = "too few text components to estimate an angle"

    # Components taller than this fraction of the page height, or wider than
    # this fraction of its width, are not glyphs (margins, stains, rules).
    _MAX_COMPONENT_HEIGHT = 0.05
    _MAX_COMPONENT_WIDTH = 0.25
    _MIN_COMPONENT_AREA = 3
    _MIN_INK_PIXELS = 50

    def __init__(
        self,
        max_angle: float = 15.0,
        min_correction: float = 0.1,
        work_size: int = 1200,
        coarse_step: float = 0.5,
        fine_step: float = 0.05,
    ) -> None:
        super().__init__(max_angle=max_angle, min_correction=min_correction)
        self.work_size = work_size
        self.coarse_step = coarse_step
        self.fine_step = fine_step

    def _ink_points(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Centered (x, y) coordinates of glyph-sized ink pixels, or None."""
        scale = min(1.0, self.work_size / max(gray.shape))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Flatten the paper before a global threshold: on a raw yellowed scan
        # Otsu can split paper from white margins/fill instead of ink from paper.
        normalizer = BackgroundNormalizer()
        background = normalizer.estimate_background(gray, normalizer._resolve_kernel(gray.shape))
        flat = np.clip(gray.astype(np.float32) / background * 255.0, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = binary.shape
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = (
            (stats[:, cv2.CC_STAT_HEIGHT] < self._MAX_COMPONENT_HEIGHT * h)
            & (stats[:, cv2.CC_STAT_WIDTH] < self._MAX_COMPONENT_WIDTH * w)
            & (stats[:, cv2.CC_STAT_AREA] >= self._MIN_COMPONENT_AREA)
        )
        keep[0] = False  # background label
        ys, xs = np.nonzero(keep[labels])
        if len(ys) < self._MIN_INK_PIXELS:
            return None
        return xs - w / 2, ys - h / 2

    @staticmethod
    def _sharpness(xs: np.ndarray, ys: np.ndarray, angle: float) -> float:
        """Sum of squared differences of the row profile after rotating by angle."""
        theta = np.radians(angle)
        # Row of each point once the page is rotated counter-clockwise by angle.
        rows = ys * np.cos(theta) - xs * np.sin(theta)
        profile = np.bincount((rows - rows.min()).astype(np.int64)).astype(np.float64)
        return float(np.sum(np.diff(profile) ** 2))

    def estimate_angle(self, gray: np.ndarray) -> float | None:
        """Estimate skew in degrees (positive = counter-clockwise), or None."""
        points = self._ink_points(gray)
        if points is None:
            return None
        xs, ys = points

        def search(center: float, half: float, step: float) -> float:
            angles = np.arange(center - half, center + half + step / 2, step)
            scores = [self._sharpness(xs, ys, a) for a in angles]
            return float(angles[int(np.argmax(scores))])

        # Rotating the page by the best angle levels it, so that angle is the
        # correction; report it with the same sign convention as HoughDeskew.
        best = search(0.0, min(2 * self.max_angle, 45.0), self.coarse_step)
        best = search(best, self.coarse_step, self.fine_step)
        return round(best, 3)
