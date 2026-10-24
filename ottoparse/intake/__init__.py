"""Intake: document routing, PDF text extraction, and rasterization (Phase 1.1)."""

from ottoparse.intake.pdf_text import extract_page_text, normalize_text
from ottoparse.intake.rasterize import rasterize_page
from ottoparse.intake.router import DefaultRouter
from ottoparse.intake.script import arabic_ratio, is_arabic_char

__all__ = [
    "DefaultRouter",
    "extract_page_text",
    "normalize_text",
    "rasterize_page",
    "arabic_ratio",
    "is_arabic_char",
]
