"""Input/output helpers for the pipeline.

Kept separate from the orchestration logic so loading an image file and writing
a cleaned page are testable in isolation and easy to swap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# File extensions treated as plain raster images (everything else that is not a
# PDF is rejected with a clear error).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def load_image(path: Path) -> np.ndarray:
    """Load a raster image file into a ``uint8`` NumPy array.

    Returns a 2-D array for grayscale images or 3-D (H, W, C) for color.
    """
    import cv2

    # cv2.imread does not accept Path on all builds; pass a string. IMREAD_UNCHANGED
    # preserves the original channel layout so to_grayscale can decide later.
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise ValueError(f"could not decode image: {path}")
    # OpenCV loads color as BGR; convert to RGB for consistency with rasterize.
    if data.ndim == 3 and data.shape[2] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    elif data.ndim == 3 and data.shape[2] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)
    return data


def write_image(image: np.ndarray, path: Path) -> Path:
    """Write an image array to ``path`` (creating parent dirs). Returns the path."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    out = image
    # Convert RGB(A) back to BGR(A) for cv2.imwrite's expectations.
    if out.ndim == 3 and out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    elif out.ndim == 3 and out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)
    if not cv2.imwrite(str(path), out):
        raise OSError(f"failed to write image: {path}")
    return path
