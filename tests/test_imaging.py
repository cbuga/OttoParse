"""Tests for the image path: deskew, denoise, binarizers, chaining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ottoparse.imaging.binarize import (
    AdaptiveGaussianBinarizer,
    OtsuBinarizer,
    SauvolaBinarizer,
)
from ottoparse.imaging.denoise import NlMeansDenoiser
from ottoparse.imaging.deskew import HoughDeskew
from ottoparse.imaging.pipeline import run_image_steps
from ottoparse.imaging.utils import to_grayscale
from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord, Provenance, SourcePath


def _record() -> PageRecord:
    return PageRecord(provenance=Provenance(Path("x.pdf"), 0, SourcePath.IMAGE, "img"))


# --- utils ---

def test_to_grayscale_from_rgb():
    rgb = np.zeros((10, 10, 3), np.uint8)
    rgb[:, :, 0] = 255  # red
    gray = to_grayscale(rgb)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8


# --- deskew ---

@pytest.mark.parametrize("angle", [5.0, -3.0, 8.0])
def test_deskew_estimates_skew(text_image, angle):
    img = text_image(angle=angle, noise_std=20.0)
    est = HoughDeskew().estimate_angle(img)
    assert est is not None
    # image-y points down, so the estimate has opposite sign to the applied rot
    assert est == pytest.approx(-angle, abs=1.5)


def test_deskew_applies_and_logs(text_image):
    rec = _record()
    img = text_image(angle=5.0, noise_std=20.0)
    HoughDeskew().apply(img, rec)
    step = next(s for s in rec.history if s.name == "deskew")
    assert step.params["applied"] is True


# --- denoise detectors ---

def test_denoise_flags_low_contrast(text_image):
    rec = _record()
    faded = np.full((300, 400), 128, np.uint8)  # flat grey = no contrast
    NlMeansDenoiser().apply(faded, rec)
    assert any(f.code is FlagCode.LOW_CONTRAST for f in rec.flags)


# --- binarizers ---

@pytest.mark.parametrize(
    "binarizer",
    [SauvolaBinarizer(), OtsuBinarizer(), AdaptiveGaussianBinarizer()],
)
def test_binarizers_produce_binary(text_image, binarizer):
    rec = _record()
    out = binarizer.apply(text_image(noise_std=15.0), rec)
    uniq = np.unique(out)
    assert set(uniq.tolist()).issubset({0, 255})
    assert out.dtype == np.uint8


def test_sauvola_rejects_even_window():
    with pytest.raises(ValueError):
        SauvolaBinarizer(window_size=24)


# --- chaining & error isolation ---

def test_run_image_steps_full_chain(text_image):
    rec = _record()
    out = run_image_steps(
        text_image(angle=4.0, noise_std=20.0),
        [HoughDeskew(), NlMeansDenoiser(), SauvolaBinarizer()],
        rec,
    )
    assert set(np.unique(out).tolist()).issubset({0, 255})
    assert not rec.has_errors
    assert [s.name for s in rec.history] == ["deskew", "denoise", "binarize"]


def test_run_image_steps_isolates_step_errors(text_image):
    class Boom:
        name = "boom"

        def apply(self, image, record):
            raise RuntimeError("kaboom")

    rec = _record()
    out = run_image_steps(text_image(), [Boom(), OtsuBinarizer()], rec)
    # the boom error is flagged, but otsu still ran and produced binary output
    assert any(f.severity is Severity.ERROR for f in rec.flags)
    assert set(np.unique(out).tolist()).issubset({0, 255})
