"""Tests for the image path: deskew, denoise, binarizers, chaining."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ottoparse.imaging.binarize import (
    AdaptiveGaussianBinarizer,
    OtsuBinarizer,
    SauvolaBinarizer,
)
from ottoparse.imaging.denoise import NlMeansDenoiser
from ottoparse.imaging.deskew import HoughDeskew, ProjectionDeskew
from ottoparse.imaging.pipeline import run_image_steps
from ottoparse.imaging.utils import to_grayscale
from ottoparse.io import load_image
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


@pytest.mark.parametrize("angle", [5.0, -3.0, 8.0])
def test_projection_deskew_estimates_skew(text_image, angle):
    img = text_image(angle=angle, noise_std=20.0)
    est = ProjectionDeskew().estimate_angle(img)
    assert est == pytest.approx(-angle, abs=0.5)


@pytest.mark.parametrize("angle", [4.0, -3.0, 7.0])
def test_projection_deskew_on_rotated_periodical_page(periodical_page, angle):
    # Yellowed RGB paper, stain, dark margin, white rotation corners, and
    # broken Ottoman-like lines: the estimate must still recover the rotation.
    rec = _record()
    page = periodical_page(angle=angle)
    est = ProjectionDeskew().estimate_angle(to_grayscale(page))
    assert est == pytest.approx(-angle, abs=0.3)
    out = ProjectionDeskew().apply(page, rec)
    assert out.shape == page.shape
    step = next(s for s in rec.history if s.name == "deskew")
    assert step.strategy == "projection"
    assert step.params["applied"] is True
    assert not any(f.code is FlagCode.SKEW_UNDETECTED for f in rec.flags)
    # The levelled page is (near) level.
    assert abs(ProjectionDeskew().estimate_angle(to_grayscale(out))) <= 0.3


def test_projection_deskew_flags_blank_page():
    rec = _record()
    ProjectionDeskew().apply(np.full((400, 300), 255, np.uint8), rec)
    assert any(f.code is FlagCode.SKEW_UNDETECTED for f in rec.flags)


_SAMPLES = sorted((Path(__file__).parents[1] / "samples").glob("1-*.jpg"))


@pytest.mark.skipif(not _SAMPLES, reason="private archival samples not present")
@pytest.mark.parametrize("path", _SAMPLES, ids=lambda p: p.name)
def test_projection_deskew_on_real_samples(path):
    # Regression: HoughDeskew flagged SKEW_UNDETECTED on every real scan.
    gray = to_grayscale(load_image(path))
    est = ProjectionDeskew().estimate_angle(gray)
    assert est is not None and abs(est) < 3.0


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


def _foxed_page() -> np.ndarray:
    """A flattened page: dark text, a faint foxing spot beside it, a faint alif."""
    page = np.full((300, 400), 245, np.uint8)
    for y in range(60, 260, 40):
        cv2.putText(page, "matbu line", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 60, 2)
    cv2.circle(page, (230, 45), 5, 130, -1)   # foxing spot just above a line
    page[150:190, 300:302] = 130              # thin, faint upright stroke
    return page


def test_otsu_drops_faint_foxing_keeps_text_and_upright_strokes():
    rec = _record()
    page = _foxed_page()
    out = OtsuBinarizer().apply(page, rec)
    assert (out[40:51, 225:236] == 255).all()        # foxing spot removed
    assert (out[150:190, 300:302] == 0).mean() > 0.9  # faint alif kept
    assert (out[page == 60] == 0).mean() > 0.99       # text kept
    step = next(s for s in rec.history if s.name == "binarize")
    assert step.strategy == "otsu_hysteresis"
    assert step.params["removed_faint"] >= 1


def test_otsu_plain_mode_keeps_foxing():
    out = OtsuBinarizer(core_contrast=None).apply(_foxed_page(), _record())
    assert (out[40:51, 225:236] == 0).any()


@pytest.mark.parametrize("kwargs", [{"core_contrast": 0.0}, {"core_contrast": 1.0},
                                    {"upright_aspect": 0}])
def test_otsu_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        OtsuBinarizer(**kwargs)


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
