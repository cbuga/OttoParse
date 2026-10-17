"""Tests for the intake path: script detection, normalization, routing."""

from __future__ import annotations

import pymupdf

from ottoparse.intake.pdf_text import normalize_text
from ottoparse.intake.router import DefaultRouter
from ottoparse.intake.script import arabic_ratio, is_arabic_char
from ottoparse.model.record import SourcePath

# --- script detection ---

def test_is_arabic_char():
    assert is_arabic_char("ع")
    assert not is_arabic_char("a")


def test_arabic_ratio():
    assert arabic_ratio("عثمانلى") == 1.0
    assert arabic_ratio("hello") == 0.0
    assert arabic_ratio("") == 0.0
    # non-letters are ignored
    assert arabic_ratio("عثمانلى 12345 !!!") == 1.0


# --- normalization ---

def test_normalize_rejoins_hyphenated_words():
    assert normalize_text("Osman-\nli") == "Osmanli"


def test_normalize_collapses_whitespace_and_blank_lines():
    out = normalize_text("a   b\t\tc\n\n\n\nd")
    assert out == "a b c\n\nd"


def test_normalize_strips_control_chars():
    assert "\x07" not in normalize_text("a\x07b")


# --- routing ---

def _one_page_pdf(text: str | None, draw: bool = False):
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 100), text)
    if draw:
        page.draw_rect(pymupdf.Rect(60, 60, 400, 500), fill=(0, 0, 0))
    return doc, page


def test_router_routes_textless_page_to_image():
    _, page = _one_page_pdf(text=None, draw=True)
    decision = DefaultRouter().decide(page)
    assert decision.path is SourcePath.IMAGE
    assert "thin" in decision.reason or "absent" in decision.reason


def test_router_flags_latin_text_as_suspect_under_arabic_threshold():
    # Enough chars, but Latin -> suspect under the default arabic threshold.
    _, page = _one_page_pdf("plenty of latin words here " * 4)
    decision = DefaultRouter(min_chars=50, min_arabic_ratio=0.30).decide(page)
    assert decision.path is SourcePath.IMAGE
    assert "suspect" in decision.reason


def test_router_accepts_text_when_arabic_threshold_disabled():
    _, page = _one_page_pdf("plenty of latin words here " * 4)
    decision = DefaultRouter(min_chars=50, min_arabic_ratio=0.0).decide(page)
    assert decision.path is SourcePath.TEXT_LAYER
