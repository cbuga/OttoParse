"""End-to-end tests for the top-level Pipeline orchestrator."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ottoparse import Pipeline
from ottoparse.intake.router import DefaultRouter
from ottoparse.model.record import SourcePath


def _pipeline(**kwargs) -> Pipeline:
    # accept any script so the synthetic Latin text page routes to TEXT_LAYER
    return Pipeline(router=DefaultRouter(min_chars=50, min_arabic_ratio=0.0), **kwargs)


def test_process_mixed_pdf_fans_out_per_page(mixed_pdf):
    records = _pipeline().process(mixed_pdf)
    assert len(records) == 2

    text_page, image_page = records
    assert text_page.provenance.source_path is SourcePath.TEXT_LAYER
    assert text_page.text and "clean embedded text layer" in text_page.text
    assert text_page.image is None

    assert image_page.provenance.source_path is SourcePath.IMAGE
    assert image_page.text is None
    assert image_page.image is not None
    assert set(np.unique(image_page.image).tolist()).issubset({0, 255})


def test_process_plain_png(skewed_png):
    records = _pipeline().process(skewed_png)
    assert len(records) == 1
    rec = records[0]
    assert rec.provenance.source_path is SourcePath.IMAGE
    assert [s.name for s in rec.history] == [
        "deskew", "enhance_contrast", "denoise",
        "confidence_map", "binarize", "artifact_filter",
    ]
    assert not rec.has_errors


def test_write_images_to_disk(skewed_png, tmp_path):
    out_dir = tmp_path / "out"
    records = _pipeline(write_images_to=out_dir).process(skewed_png)
    rec = records[0]
    assert rec.image_path is not None
    assert rec.image_path.exists()


def test_records_are_json_serializable(mixed_pdf):
    records = _pipeline().process(mixed_pdf)
    blob = json.dumps([r.to_dict() for r in records])  # must not raise
    assert len(json.loads(blob)) == 2


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        _pipeline().process("does_not_exist.pdf")


def test_unsupported_type_raises(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError):
        _pipeline().process(bad)
