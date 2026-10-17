"""Denoising & noise removal (diagram step 1.3).

Archival scans carry speckle, scanner noise, and often verso ink showing
through thin paper (bleed-through) or faded, low-contrast printing. This step
smooths noise while preserving edges, and inspects the page to raise the
quality flags the plan calls out: ``HIGH_NOISE``, ``LOW_CONTRAST``, and
``BLEED_THROUGH``.

The detectors are deliberately cheap, measurement-based signals — they annotate
the record so the downstream "confidence filtering" stage has numbers to
threshold on; they do not attempt to *fix* bleed-through (that is binarization's
job and, for hard cases, a Phase 2 concern).
"""

from __future__ import annotations

import cv2
import numpy as np

from ottoparse.imaging.utils import to_grayscale
from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord


class NlMeansDenoiser:
    """Edge-preserving denoiser (fast Non-Local Means) with quality detectors.

    Args:
        strength: Filter strength ``h`` for Non-Local Means. Higher removes more
            noise but risks softening fine print; 10 is a balanced default.
        noise_warn: Std-dev (0-255) of a flat background patch above which the
            page is flagged ``HIGH_NOISE``.
        contrast_warn: Dynamic-range span (0-255) below which the page is
            flagged ``LOW_CONTRAST`` (faded print).
        bleed_warn: Fraction of mid-tone pixels above which the page is flagged
            ``BLEED_THROUGH`` (verso ink shows as a grey haze between the white
            background and the black foreground).
    """

    name = "denoise"

    def __init__(
        self,
        strength: float = 10.0,
        noise_warn: float = 18.0,
        contrast_warn: float = 60.0,
        bleed_warn: float = 0.12,
    ) -> None:
        self.strength = strength
        self.noise_warn = noise_warn
        self.contrast_warn = contrast_warn
        self.bleed_warn = bleed_warn

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Std-dev of the brightest (background) band — a flat-field noise proxy."""
        thresh = np.percentile(gray, 85)
        background = gray[gray >= thresh]
        return float(background.std()) if background.size else 0.0

    def _contrast_span(self, gray: np.ndarray) -> float:
        """Robust dynamic range: 95th minus 5th percentile."""
        lo, hi = np.percentile(gray, [5, 95])
        return float(hi - lo)

    def _bleed_fraction(self, gray: np.ndarray) -> float:
        """Fraction of pixels in the mid-tone band (grey haze from verso ink)."""
        mid = (gray > 90) & (gray < 170)
        return float(mid.mean())

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        gray = to_grayscale(image)

        noise = self._estimate_noise(gray)
        contrast = self._contrast_span(gray)
        bleed = self._bleed_fraction(gray)

        if noise > self.noise_warn:
            record.add_flag(
                FlagCode.HIGH_NOISE, Severity.WARNING, measure=round(noise, 2),
                detail="background noise above threshold", step=self.name,
            )
        if contrast < self.contrast_warn:
            record.add_flag(
                FlagCode.LOW_CONTRAST, Severity.WARNING, measure=round(contrast, 2),
                detail="faded / low dynamic range", step=self.name,
            )
        if bleed > self.bleed_warn:
            record.add_flag(
                FlagCode.BLEED_THROUGH, Severity.WARNING, measure=round(bleed, 3),
                detail="high mid-tone fraction (possible verso bleed-through)",
                step=self.name,
            )

        denoised = cv2.fastNlMeansDenoising(gray, None, h=self.strength,
                                            templateWindowSize=7, searchWindowSize=21)
        record.add_step(
            self.name, "nl_means", strength=self.strength,
            noise=round(noise, 2), contrast=round(contrast, 2), bleed=round(bleed, 3),
        )
        return denoised
