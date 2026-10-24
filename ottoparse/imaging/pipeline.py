"""Chain image-cleanup steps into one pass over a page image.

:func:`run_image_steps` applies an ordered list of :class:`ImageStep`s to an
image, threading the same :class:`PageRecord` through so each step can log
history and flags. Per the project's "never silently drop" rule, a step that
raises is caught: the error becomes a ``PROCESSING_ERROR`` flag and the pipeline
continues from the last good image rather than aborting the page.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ottoparse.core.strategy import ImageStep
from ottoparse.imaging.utils import to_grayscale
from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord

#: Steps with this name are binarizers; the image feeding the first one is the
#: lossless grayscale we preserve on the record.
BINARIZE_STEP_NAME = "binarize"


def run_image_steps(
    image: np.ndarray,
    steps: Sequence[ImageStep],
    record: PageRecord,
) -> np.ndarray:
    """Apply ``steps`` to ``image`` in order, returning the final image.

    Each step receives the output of the previous one. If a step raises, the
    exception is recorded as a ``PROCESSING_ERROR`` flag (ERROR severity) and
    the current image is passed unchanged to the next step.

    Grayscale preservation: the cleaned image feeding the **first binarizer** is
    snapshotted to ``record.gray`` (as grayscale) before the irreversible
    binarization runs. Binarization drops every sub-threshold stroke — the faint
    text under a stain — so the grayscale is kept for downstream reconstruction.
    If there is no binarizer, the final cleaned image is stored as ``gray``.
    """
    current = image
    captured_gray = False
    for step in steps:
        # Snapshot the grayscale just before the first binarization.
        if getattr(step, "name", None) == BINARIZE_STEP_NAME and not captured_gray:
            record.gray = to_grayscale(current).copy()
            captured_gray = True
        try:
            current = step.apply(current, record)
        except Exception as exc:  # noqa: BLE001 - we must not abort the page
            record.add_flag(
                FlagCode.PROCESSING_ERROR, Severity.ERROR,
                detail=f"{type(exc).__name__} in step '{step.name}': {exc}",
                step=step.name,
            )
    # No binarizer ran: the cleaned grayscale is itself the output.
    if not captured_gray and current is not None:
        record.gray = to_grayscale(current).copy()
    return current
