"""Tests for the data model: records, flags, provenance, serialization."""

from __future__ import annotations

import json
from pathlib import Path

from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord, Provenance, SourcePath


def _record(path: SourcePath = SourcePath.IMAGE) -> PageRecord:
    return PageRecord(provenance=Provenance(Path("a.pdf"), 0, path, "reason"))


def test_is_text_vs_image():
    assert _record(SourcePath.TEXT_LAYER).is_text
    assert _record(SourcePath.IMAGE).is_image


def test_add_flag_and_step():
    rec = _record()
    rec.add_step("deskew", "hough", angle=1.2)
    flag = rec.add_flag(FlagCode.SEVERE_SKEW, Severity.WARNING, measure=7.5, step="deskew")
    assert flag in rec.flags
    assert rec.history[0].name == "deskew"
    assert rec.history[0].params == {"angle": 1.2}


def test_has_errors_and_max_severity():
    rec = _record()
    assert rec.max_severity is None
    assert not rec.has_errors
    rec.add_flag(FlagCode.LOW_CONTRAST, Severity.INFO)
    rec.add_flag(FlagCode.HIGH_NOISE, Severity.WARNING)
    assert rec.max_severity is Severity.WARNING
    assert not rec.has_errors
    rec.add_flag(FlagCode.PROCESSING_ERROR, Severity.ERROR)
    assert rec.max_severity is Severity.ERROR
    assert rec.has_errors


def test_to_dict_is_json_serializable():
    rec = _record(SourcePath.TEXT_LAYER)
    rec.text = "some text"
    rec.add_flag(FlagCode.SUSPECT_TEXT_LAYER, Severity.WARNING, measure=0.1)
    rec.add_step("normalize_text", "regex")
    blob = json.dumps(rec.to_dict())  # must not raise
    parsed = json.loads(blob)
    assert parsed["text"] == "some text"
    assert parsed["provenance"]["source_path"] == "text_layer"
    assert parsed["flags"][0]["code"] == "SUSPECT_TEXT_LAYER"


def test_flagcode_is_str():
    # StrEnum members behave as plain strings for JSON / comparisons.
    assert FlagCode.SEVERE_SKEW == "SEVERE_SKEW"
    assert json.dumps({"c": FlagCode.SEVERE_SKEW}) == '{"c": "SEVERE_SKEW"}'
