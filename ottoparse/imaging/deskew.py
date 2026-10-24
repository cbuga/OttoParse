"""Deskewing & alignment (diagram step 1.2).

Printed pages are often scanned at a small rotation. Left uncorrected, skew
hurts line detection and recognition downstream. :class:`HoughDeskew` estimates
the dominant skew angle and rotates the page to level it.

The estimator uses the probabilistic Hough transform over edge pixels: text
baselines and rules produce many near-horizontal line segments whose median
angle is a robust estimate of page skew. If the estimated angle exceeds a safe
range the page is still rotated best-effort but flagged ``SEVERE_SKEW``; if no
lines are found the page is left as-is and flagged ``SKEW_UNDETECTED``.
"""

from __future__ import annotations

import cv2
import numpy as np

from ottoparse.imaging.utils import to_grayscale
from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord


class HoughDeskew:
    """Estimate and correct page skew via the probabilistic Hough transform.

    Args:
        max_angle: Angles beyond +/- this (degrees) are treated as unreliable;
            the page is flagged ``SEVERE_SKEW`` and the correction is clamped.
        min_correction: Angles with magnitude below this (degrees) are ignored;
            the page is considered already level (avoids needless resampling).
    """

    name = "deskew"

    def __init__(self, max_angle: float = 15.0, min_correction: float = 0.1) -> None:
        self.max_angle = max_angle
        self.min_correction = min_correction

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
                detail="no near-horizontal lines found", step=self.name,
            )
            record.add_step(self.name, "hough", angle=None, applied=False)
            return image

        if abs(angle) > self.max_angle:
            record.add_flag(
                FlagCode.SEVERE_SKEW, Severity.WARNING, measure=abs(angle),
                detail=f"estimated {angle:.2f} deg exceeds +/-{self.max_angle}",
                step=self.name,
            )
            angle = float(np.clip(angle, -self.max_angle, self.max_angle))

        if abs(angle) < self.min_correction:
            record.add_step(self.name, "hough", angle=angle, applied=False)
            return image

        rotated = self._rotate(image, angle)
        record.add_step(self.name, "hough", angle=round(angle, 3), applied=True)
        return rotated
