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


def _periodical_page(angle: float = 0.0, size=(1450, 1000), seed: int = 0) -> np.ndarray:
    """An RGB page mimicking a late-Ottoman periodical scan, optionally rotated.

    Yellowed paper, a dark scanner margin, a brown stain, and two columns split
    by a rule. Each line is a run of baseline-aligned "words" of uneven height
    with dots above and below and wide gaps between them, so there is no long
    straight baseline -- like Ottoman Turkish print, unlike ``cv2.putText``. The
    rotation fills the corners with white, as a scanner bed or a prior rotation
    would.
    """
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (196, 174, 132)
    # Ragged dark scanner margin along the left edge (no clean straight border).
    for y in range(h):
        img[y, : 20 + int(rng.integers(0, 25))] = 45
    mid = w // 2
    img[80 : h - 80, mid - 1 : mid + 1] = 60  # column rule
    for x0, x1 in ((80, mid - 25), (mid + 25, w - 80)):
        for base in range(110, h - 90, 34):
            x = x0
            while x < x1 - 40:
                word = int(rng.integers(25, 110))
                end = min(x + word, x1)
                for px in range(x, end, 22):  # disconnected letter groups
                    cv2.rectangle(img, (px, base - 3), (min(px + 16, end), base),
                                  (30, 30, 30), -1)
                for sx in range(x, end, 9):  # uneven letter heights on the baseline
                    top = base - int(rng.integers(4, 16))
                    cv2.line(img, (sx, top), (sx, base), (30, 30, 30), 2)
                for _ in range(int(rng.integers(1, 4))):  # nokta dots
                    dx = int(rng.integers(x, end))
                    dy = base + 6 if rng.random() < 0.5 else base - 20
                    cv2.circle(img, (dx, dy), 2, (30, 30, 30), -1)
                x = end + int(rng.integers(12, 35))
    yy, xx = np.mgrid[:h, :w]
    stain = (yy - 900) ** 2 / 150**2 + (xx - 300) ** 2 / 120**2 <= 1.0
    img[stain] = (img[stain] * np.array([0.65, 0.55, 0.45])).astype(np.uint8)
    if angle:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
    return img


@pytest.fixture
def periodical_page():
    """Factory for synthetic, real-like Ottoman Turkish periodical pages (RGB)."""
    return _periodical_page


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
