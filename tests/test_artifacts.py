"""Tests for artifact removal: ComponentFilter and RegionMask."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ottoparse.imaging.artifacts import ComponentFilter, RegionMask
from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord, Provenance, SourcePath


def _record() -> PageRecord:
    return PageRecord(provenance=Provenance(Path("x.jpg"), 0, SourcePath.IMAGE, "img"))


def _binary_with_blob_and_speck() -> np.ndarray:
    """White page (255) with a text stroke, a solid blob, and a 1px speck."""
    img = np.full((200, 200), 255, np.uint8)
    # thin text-like stroke (not a blob: low fill in its bbox)
    cv2.line(img, (20, 20), (180, 22), 0, 1)
    # solid filled square = stamp/blob
    cv2.rectangle(img, (60, 80), (110, 130), 0, thickness=-1)
    # a single-pixel speck
    img[160, 160] = 0
    return img


# --- ComponentFilter ---

def test_component_filter_removes_blob_and_speck():
    rec = _record()
    img = _binary_with_blob_and_speck()
    out = ComponentFilter(min_area=6, max_area=900, blob_fill=0.45).apply(img, rec)
    step = next(s for s in rec.history if s.name == "artifact_filter")
    assert step.params["removed_blobs"] >= 1
    assert step.params["removed_specks"] >= 1
    # the solid square region should now be background
    assert out[105, 85] == 255
    assert any(f.code is FlagCode.ARTIFACT_REMOVED for f in rec.flags)


def test_component_filter_keeps_output_binary():
    rec = _record()
    out = ComponentFilter().apply(_binary_with_blob_and_speck(), rec)
    assert set(np.unique(out).tolist()).issubset({0, 255})


def test_component_filter_skips_non_binary_gracefully():
    rec = _record()
    gray = np.random.default_rng(0).integers(0, 256, (50, 50), dtype=np.uint8)
    out = ComponentFilter().apply(gray, rec)
    # passed through unchanged, with a warning flag, no crash
    assert out is gray
    assert any(f.severity is Severity.WARNING for f in rec.flags)


# --- RegionMask ---

def test_region_mask_blanks_top_band_and_flags():
    rec = _record()
    img = np.zeros((100, 100), np.uint8)  # all ink
    out = RegionMask(top=0.2).apply(img, rec)
    assert (out[:20] == 255).all()   # top 20% blanked
    assert (out[20:] == 0).all()     # rest untouched
    assert any(f.code is FlagCode.REGION_MASKED for f in rec.flags)


def test_region_mask_all_sides():
    rec = _record()
    img = np.zeros((100, 100), np.uint8)
    out = RegionMask(top=0.1, bottom=0.1, left=0.1, right=0.1).apply(img, rec)
    assert (out[:10] == 255).all()
    assert (out[-10:] == 255).all()
    assert (out[:, :10] == 255).all()
    assert (out[:, -10:] == 255).all()


def test_region_mask_rejects_out_of_range():
    import pytest
    with pytest.raises(ValueError):
        RegionMask(top=1.0)
