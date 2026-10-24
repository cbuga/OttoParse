"""Artifact removal: seals, stamps, ink blots, ghost specks, and region masks.

Historical periodicals carry non-body material — circular seals and stamps,
editorial notes, a central title emblem/watermark, and ink blots — that survive
binarization as blobs and fragmented ghosts and pollute downstream line
serialization. Two complementary, opt-in :class:`ImageStep`s address this:

- :class:`ComponentFilter` — operates on a binary image, removes connected
  components whose *shape* matches an artifact signature (tiny specks; large
  solid blobs such as stamps and ink blots) while leaving text alone. Robust to
  where the artifact sits on the page.
- :class:`RegionMask` — blanks configurable page regions expressed as
  *fractions* of page dimensions (e.g. the top masthead band and side margins),
  for cases where a hard spatial exclusion is wanted. It records the masked
  bounds and raises a flag, so nothing is removed silently.

Both are **off by default**: they are layout-specific and lossy, so a caller
opts in by adding them to ``image_steps``. Both expect/produce the project's
binary convention: foreground (ink) = 0, background = 255.
"""

from __future__ import annotations

import cv2
import numpy as np

from ottoparse.imaging.utils import is_binary
from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord


class ComponentFilter:
    """Remove artifact-shaped connected components from a binary page.

    Works on the foreground (ink) components. A component is removed if it looks
    like a speck or a solid blob rather than text:

    - **specks**: area <= ``min_area`` pixels (isolated ghost fragments);
    - **blobs**: area >= ``max_area`` pixels AND ``fill_ratio`` >= ``blob_fill``,
      where fill_ratio is the component's pixel count divided by its bounding-box
      area. Stamps, seals and ink blots are large and densely filled; text —
      even dense Ottoman print — rarely fills its bounding box so solidly at large size.

    Thresholds are defaulted from analysis of a real periodical scan but are
    meant to be tuned per corpus. All removals are counted into the record.

    Args:
        min_area: Components with area <= this are dropped as specks. Set to 0
            to disable speck removal.
        max_area: Minimum area for a component to be *considered* a blob.
        blob_fill: Minimum bounding-box fill ratio for a large component to be
            treated as a blob (0-1).
        connectivity: 4 or 8 connectivity for component labelling.
    """

    name = "artifact_filter"

    def __init__(
        self,
        min_area: int = 6,
        max_area: int = 900,
        blob_fill: float = 0.45,
        connectivity: int = 8,
    ) -> None:
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self.min_area = min_area
        self.max_area = max_area
        self.blob_fill = blob_fill
        self.connectivity = connectivity

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        if not is_binary(image):
            # Needs a binary image; flag and pass through rather than guessing.
            record.add_flag(
                FlagCode.PROCESSING_ERROR, Severity.WARNING,
                detail="ComponentFilter expects a binary image; place it after "
                "binarization. Skipped.",
                step=self.name,
            )
            return image

        # Foreground mask (ink = 0 -> 1).
        fg = (image == 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            fg, connectivity=self.connectivity
        )

        remove = np.zeros(n, dtype=bool)
        specks = blobs = 0
        for label in range(1, n):  # 0 is background
            area = int(stats[label, cv2.CC_STAT_AREA])
            if self.min_area and area <= self.min_area:
                remove[label] = True
                specks += 1
                continue
            if area >= self.max_area:
                bw = int(stats[label, cv2.CC_STAT_WIDTH])
                bh = int(stats[label, cv2.CC_STAT_HEIGHT])
                fill = area / float(bw * bh) if bw and bh else 0.0
                if fill >= self.blob_fill:
                    remove[label] = True
                    blobs += 1

        if specks or blobs:
            remove_mask = remove[labels]  # bool image of pixels to clear
            cleaned = image.copy()
            cleaned[remove_mask] = 255  # set removed ink to background
        else:
            cleaned = image

        record.add_step(
            self.name, "connected_components",
            min_area=self.min_area, max_area=self.max_area,
            blob_fill=self.blob_fill,
            removed_specks=specks, removed_blobs=blobs,
        )
        if blobs:
            record.add_flag(
                FlagCode.ARTIFACT_REMOVED, Severity.INFO, measure=float(blobs),
                detail=f"removed {blobs} blob-like component(s) "
                f"(stamps/seals/ink blots)",
                step=self.name,
            )
        return cleaned


class RegionMask:
    """Blank configurable page regions, expressed as fractions of page size.

    Useful for hard spatial exclusion of a masthead band or side margins on
    periodicals, where seals/emblems/editorial notes concentrate. Because this
    is lossy (it may also cover legitimate masthead metadata), it records the
    masked bounds and raises a WARNING flag so the removal is auditable.

    Fractions are of the current image dimensions, so the step is resolution-
    and layout-agnostic. All default to 0 (no masking) — the caller opts in.

    Args:
        top: Fraction of height masked from the top (0-1).
        bottom: Fraction of height masked from the bottom (0-1).
        left: Fraction of width masked from the left (0-1).
        right: Fraction of width masked from the right (0-1).
    """

    name = "region_mask"

    def __init__(
        self, top: float = 0.0, bottom: float = 0.0,
        left: float = 0.0, right: float = 0.0,
    ) -> None:
        for n, v in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{n} must be in [0.0, 1.0)")
        self.top, self.bottom, self.left, self.right = top, bottom, left, right

    def apply(self, image: np.ndarray, record: PageRecord) -> np.ndarray:
        h, w = image.shape[:2]
        # Background value matches the image's convention (255 for binary/gray).
        bg = 255
        out = image.copy()

        masked_any = False
        if self.top:
            out[: int(h * self.top)] = bg
            masked_any = True
        if self.bottom:
            out[h - int(h * self.bottom):] = bg
            masked_any = True
        if self.left:
            out[:, : int(w * self.left)] = bg
            masked_any = True
        if self.right:
            out[:, w - int(w * self.right):] = bg
            masked_any = True

        record.add_step(
            self.name, "fractional_bounds",
            top=self.top, bottom=self.bottom, left=self.left, right=self.right,
        )
        if masked_any:
            record.add_flag(
                FlagCode.REGION_MASKED, Severity.WARNING,
                detail=f"masked regions t={self.top} b={self.bottom} "
                f"l={self.left} r={self.right}; masthead metadata in these "
                f"bands is suppressed",
                step=self.name,
            )
        return out
