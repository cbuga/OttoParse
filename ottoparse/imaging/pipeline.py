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
from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord


def run_image_steps(
    image: np.ndarray,
    steps: Sequence[ImageStep],
    record: PageRecord,
) -> np.ndarray:
    """Apply ``steps`` to ``image`` in order, returning the final image.

    Each step receives the output of the previous one. If a step raises, the
    exception is recorded as a ``PROCESSING_ERROR`` flag (ERROR severity) and
    the current image is passed unchanged to the next step.
    """
    current = image
    for step in steps:
        try:
            current = step.apply(current, record)
        except Exception as exc:  # noqa: BLE001 - we must not abort the page
            record.add_flag(
                FlagCode.PROCESSING_ERROR, Severity.ERROR,
                detail=f"{type(exc).__name__} in step '{step.name}': {exc}",
                step=step.name,
            )
    return current
