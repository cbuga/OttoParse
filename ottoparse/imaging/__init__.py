"""Imaging: deskew, denoise, and adaptive binarization (Phase 1.2-1.4)."""

from ottoparse.imaging.artifacts import ComponentFilter, RegionMask
from ottoparse.imaging.background import BackgroundNormalizer
from ottoparse.imaging.binarize import (
    AdaptiveGaussianBinarizer,
    OtsuBinarizer,
    SauvolaBinarizer,
)
from ottoparse.imaging.confidence import ConfidenceMap
from ottoparse.imaging.denoise import NlMeansDenoiser
from ottoparse.imaging.deskew import HoughDeskew, ProjectionDeskew
from ottoparse.imaging.enhance import ClaheEnhancer
from ottoparse.imaging.pipeline import run_image_steps
from ottoparse.imaging.utils import to_grayscale

__all__ = [
    "ProjectionDeskew",
    "HoughDeskew",
    "NlMeansDenoiser",
    "SauvolaBinarizer",
    "OtsuBinarizer",
    "AdaptiveGaussianBinarizer",
    "ComponentFilter",
    "RegionMask",
    "ConfidenceMap",
    "ClaheEnhancer",
    "BackgroundNormalizer",
    "run_image_steps",
    "to_grayscale",
]
