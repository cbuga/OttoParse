"""Tests for background normalization (stain / uneven-paper flattening)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ottoparse.imaging.background import BackgroundNormalizer
from ottoparse.imaging.binarize import OtsuBinarizer, SauvolaBinarizer
from ottoparse.model.record import PageRecord, Provenance, SourcePath


def _record() -> PageRecord:
    return PageRecord(provenance=Provenance(Path("x.png"), 0, SourcePath.IMAGE, "img"))


def _stained(text_image) -> tuple[np.ndarray, np.ndarray]:
    """A text page darkened by a broad multiplicative 'stain' blob.

    Returns the stained page and a boolean mask of stain pixels that hold no
    ink (paper under the stain), where a good result must stay white.
    """
    page = text_image().astype(np.float32)
    yy, xx = np.mgrid[: page.shape[0], : page.shape[1]]
    stain = (yy - 250) ** 2 / 120**2 + (xx - 350) ** 2 / 200**2 <= 1.0
    page[stain] *= 0.6
    paper_under_stain = stain & (text_image() == 255)
    return page.astype(np.uint8), paper_under_stain


def test_flattens_stain_to_paper(text_image):
    stained, paper = _stained(text_image)
    flat = BackgroundNormalizer(kernel_size=31).apply(stained, _record())
    # Paper under the stain is lifted back to (near) white...
    assert np.median(flat[paper]) >= 240
    # ...while ink stays dark.
    ink = text_image() == 0
    assert np.median(flat[ink]) <= 80


def test_removes_stain_outline_after_threshold(text_image):
    stained, paper = _stained(text_image)
    rec = _record()
    flat = BackgroundNormalizer(kernel_size=31).apply(stained, rec)
    binary = OtsuBinarizer().apply(flat, rec)
    raw_binary = SauvolaBinarizer().apply(stained, _record())
    # Normalized: almost no false ink on the stained paper.
    assert (binary[paper] == 0).mean() < 0.01
    # And clearly better than thresholding the stained page directly.
    assert (binary[paper] == 0).mean() < (raw_binary[paper] == 0).mean()


def test_output_is_2d_uint8_from_rgb(text_image):
    gray = text_image()
    rgb = np.dstack([gray, gray, gray])
    out = BackgroundNormalizer(channel="blue").apply(rgb, _record())
    assert out.ndim == 2
    assert out.dtype == np.uint8
    assert out.shape == gray.shape


def test_channel_selection_uses_requested_channel():
    # Yellow paper (bright R/G, dim B) with a stain visible only in blue.
    rgb = np.full((200, 200, 3), (230, 220, 160), np.uint8)
    rgb[50:150, 50:150, 2] = 80
    rec = _record()
    BackgroundNormalizer(channel="blue", kernel_size=15).apply(rgb, rec)
    assert rec.history[-1].params["channel"] == "blue"


def test_grayscale_input_records_fallback(text_image):
    rec = _record()
    BackgroundNormalizer(channel="blue").apply(text_image(), rec)
    step = rec.history[-1]
    assert step.name == "normalize_background"
    assert step.params["channel"] == "gray"


def test_auto_kernel_scales_with_page_size():
    rec_small, rec_big = _record(), _record()
    BackgroundNormalizer().apply(np.full((500, 700), 255, np.uint8), rec_small)
    BackgroundNormalizer().apply(np.full((2000, 2800), 255, np.uint8), rec_big)
    small = rec_small.history[-1].params["kernel_size"]
    big = rec_big.history[-1].params["kernel_size"]
    assert small % 2 == 1 and big % 2 == 1
    assert big > small


@pytest.mark.parametrize(
    "kwargs",
    [{"channel": "purple"}, {"kernel_size": 40}, {"kernel_size": 1}, {"blur_sigma": 0}],
)
def test_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        BackgroundNormalizer(**kwargs)
