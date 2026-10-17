"""Shared pytest fixtures: synthetic images and PDFs.

These build small, deterministic inputs in a temp dir so the suite needs no
sample-data files and runs fast.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest


def _text_image(angle: float = 0.0, noise_std: float = 0.0, size=(500, 700)) -> np.ndarray:
    """A white page with black text lines, optionally rotated and noised."""
    h, w = size
    img = np.full((h, w), 255, np.uint8)
    for y in range(70, h - 40, 40):
        cv2.putText(img, "ottoman matbu scan line", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    if angle:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=255)
    if noise_std:
        rng = np.random.default_rng(0)
        img = np.clip(img.astype(int) + rng.normal(0, noise_std, img.shape), 0, 255)
        img = img.astype(np.uint8)
    return img


@pytest.fixture
def text_image():
    """Factory for synthetic text-page images."""
    return _text_image


@pytest.fixture
def skewed_png(tmp_path: Path) -> Path:
    """A plain PNG scan with a 4-degree skew and light noise."""
    path = tmp_path / "scan.png"
    cv2.imwrite(str(path), _text_image(angle=4.0, noise_std=20.0))
    return path


@pytest.fixture
def mixed_pdf(tmp_path: Path) -> Path:
    """A 2-page PDF: page 0 has a text layer, page 1 is image-only."""
    doc = pymupdf.open()
    p0 = doc.new_page()
    p0.insert_text((72, 100),
                   "A clean embedded text layer with plenty of readable words. " * 3)
    p1 = doc.new_page()
    p1.draw_rect(pymupdf.Rect(60, 60, 400, 500), fill=(0, 0, 0))
    path = tmp_path / "mixed.pdf"
    doc.save(path)
    doc.close()
    return path
