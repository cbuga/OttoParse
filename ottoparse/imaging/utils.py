"""Shared helpers for image steps."""

from __future__ import annotations

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a 2-D ``uint8`` grayscale view of ``image``.

    Accepts grayscale (H, W), RGB (H, W, 3), or RGBA (H, W, 4). RGB/RGBA are
    converted with the luminosity weighting. Already-grayscale input is returned
    unchanged (as ``uint8``).
    """
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        rgb = image[:, :, :3].astype(np.float32)
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return np.clip(gray, 0, 255).astype(np.uint8)
    raise ValueError(f"unsupported image shape for grayscale: {image.shape}")


def is_binary(image: np.ndarray) -> bool:
    """True if ``image`` contains only two distinct values (already binarized)."""
    return image.ndim == 2 and len(np.unique(image)) <= 2
