"""Rasterize PDF pages to image arrays.

Pages routed to the image path (scans, or suspect text layers) are rendered to
a raster at a chosen DPI and handed to the image-cleanup pipeline as a NumPy
array. Historical print is rendered to grayscale by default: it is what the
downstream deskew/denoise/binarize steps expect, and it halves memory versus
color with no loss for monochrome documents.
"""

from __future__ import annotations

import numpy as np

# 72 is the PDF userspace unit (points per inch); the zoom factor that yields a
# target DPI is therefore dpi / 72.
_PDF_BASE_DPI = 72.0


def rasterize_page(page: object, dpi: int = 300, grayscale: bool = True) -> np.ndarray:
    """Render a PyMuPDF page to a 2-D (grayscale) or 3-D (RGB) array.

    Args:
        page: A ``fitz.Page`` (typed loosely to avoid a hard import here).
        dpi: Target render resolution. 300 is a good default for OCR-grade
            historical print.
        grayscale: If True, render single-channel grayscale; else RGB.

    Returns:
        A ``uint8`` NumPy array: shape ``(H, W)`` if grayscale, else
        ``(H, W, 3)``.
    """
    import pymupdf  # imported lazily so the module stays light to import

    zoom = dpi / _PDF_BASE_DPI
    matrix = pymupdf.Matrix(zoom, zoom)
    colorspace = pymupdf.csGRAY if grayscale else pymupdf.csRGB
    pix = page.get_pixmap(matrix=matrix, colorspace=colorspace, alpha=False)  # type: ignore[attr-defined]

    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    if grayscale:
        return arr.reshape(pix.height, pix.width)
    return arr.reshape(pix.height, pix.width, pix.n)
