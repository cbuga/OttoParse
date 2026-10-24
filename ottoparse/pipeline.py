"""Top-level Phase 1 orchestrator.

:class:`Pipeline` is the single public entry point. Given an input file it:

1. dispatches on file type (PDF vs plain raster image);
2. for PDFs, routes **each page** to the text-layer path or the image path;
3. runs the appropriate path, producing exactly one :class:`PageRecord` per page;
4. optionally writes cleaned images to disk.

It never aborts a run because of one bad page: unreadable pages become records
carrying an ``UNREADABLE_PAGE`` error flag, so a single failure cannot take down
a multi-page batch.
"""

from __future__ import annotations

from pathlib import Path

from ottoparse.core.config import PipelineConfig
from ottoparse.core.strategy import ImageStep, PageRouter
from ottoparse.imaging.artifacts import ComponentFilter
from ottoparse.imaging.background import BackgroundNormalizer
from ottoparse.imaging.binarize import OtsuBinarizer
from ottoparse.imaging.confidence import ConfidenceMap
from ottoparse.imaging.denoise import NlMeansDenoiser
from ottoparse.imaging.deskew import ProjectionDeskew
from ottoparse.imaging.pipeline import run_image_steps
from ottoparse.intake.pdf_text import extract_page_text, normalize_text
from ottoparse.intake.rasterize import rasterize_page
from ottoparse.intake.router import DefaultRouter
from ottoparse.io import IMAGE_EXTENSIONS, PDF_EXTENSIONS, load_image, write_image
from ottoparse.model.flags import FlagCode, Severity
from ottoparse.model.record import PageRecord, Provenance, SourcePath


def default_image_steps() -> list[ImageStep]:
    """The default Phase 1 image-cleanup chain.

    Order: deskew -> background normalization -> denoise -> confidence map ->
    Otsu -> component filter.

    - ``ProjectionDeskew`` estimates skew from the sharpness of the row ink
      profile; ``HoughDeskew`` finds no long baselines on Arabic-script print.
    - ``BackgroundNormalizer`` divides the page by its estimated paper tone, so
      stains and uneven lighting flatten to white while ink keeps its contrast.
      This replaces CLAHE, which amplified stain edges into black contours.
    - Once the background is flat a single global threshold suffices, so Otsu
      is used; it keeps strokes and dots intact better than Sauvola here.
    - ``ConfidenceMap`` runs on the cleaned **grayscale** (before binarization,
      where contrast is still meaningful) to flag faint/stained regions for
      downstream reconstruction — it deletes nothing.
    - ``ComponentFilter`` runs **after** binarization to strip the ghost specks
      left after thresholding (it needs a binary image).
    """
    return [
        ProjectionDeskew(),
        BackgroundNormalizer(),
        NlMeansDenoiser(),
        ConfidenceMap(),
        OtsuBinarizer(),
        ComponentFilter(),
    ]


class Pipeline:
    """Run Phase 1 intake & pre-processing on a document.

    Args:
        router: Per-page text-vs-image router. Defaults to :class:`DefaultRouter`.
        image_steps: Ordered image-cleanup steps. Defaults to
            :func:`default_image_steps`.
        rasterize_dpi: DPI for rendering image-only PDF pages.
        write_images_to: If set, cleaned images are written here and each
            record's ``image_path`` is populated.
        config: A fully-built :class:`PipelineConfig`. If given, the other
            arguments are ignored. Useful for advanced wiring.
    """

    def __init__(
        self,
        router: PageRouter | None = None,
        image_steps: list[ImageStep] | None = None,
        rasterize_dpi: int = 300,
        write_images_to: str | Path | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = PipelineConfig(
                router=router or DefaultRouter(),
                image_steps=image_steps if image_steps is not None else default_image_steps(),
                rasterize_dpi=rasterize_dpi,
                write_images_to=Path(write_images_to) if write_images_to else None,
            )

    # -- public API -----------------------------------------------------------

    def process(self, source: str | Path) -> list[PageRecord]:
        """Process a document into a list of :class:`PageRecord`, one per page."""
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix in PDF_EXTENSIONS:
            return self._process_pdf(path)
        if suffix in IMAGE_EXTENSIONS:
            return [self._process_image_file(path)]
        raise ValueError(
            f"unsupported input type '{suffix}'. Expected a PDF or one of "
            f"{sorted(IMAGE_EXTENSIONS)}"
        )

    # -- PDF handling ---------------------------------------------------------

    def _process_pdf(self, path: Path) -> list[PageRecord]:
        import pymupdf

        records: list[PageRecord] = []
        with pymupdf.open(path) as doc:
            for page_index in range(doc.page_count):
                records.append(self._process_pdf_page(doc, page_index, path))
        return records

    def _process_pdf_page(self, doc: object, page_index: int, path: Path) -> PageRecord:
        try:
            page = doc[page_index]  # type: ignore[index]
        except Exception as exc:  # noqa: BLE001
            rec = PageRecord(
                provenance=Provenance(path, page_index, SourcePath.IMAGE, "load failed")
            )
            rec.add_flag(
                FlagCode.UNREADABLE_PAGE, Severity.ERROR,
                detail=f"{type(exc).__name__}: {exc}",
            )
            return rec

        decision = self.config.router.decide(page)

        if decision.path is SourcePath.TEXT_LAYER:
            return self._handle_text_page(page, page_index, path, decision.reason)
        return self._handle_pdf_image_page(page, page_index, path, decision.reason)

    def _handle_text_page(
        self, page: object, page_index: int, path: Path, reason: str
    ) -> PageRecord:
        rec = PageRecord(
            provenance=Provenance(path, page_index, SourcePath.TEXT_LAYER, reason)
        )
        raw = extract_page_text(page)
        rec.text = normalize_text(raw)
        rec.add_step("extract_text", "pymupdf")
        rec.add_step("normalize_text", "regex")
        return rec

    def _handle_pdf_image_page(
        self, page: object, page_index: int, path: Path, reason: str
    ) -> PageRecord:
        rec = PageRecord(
            provenance=Provenance(path, page_index, SourcePath.IMAGE, reason)
        )
        try:
            # Rasterize in color: stains and ink differ in hue, and color-aware
            # steps (e.g. BackgroundNormalizer(channel="blue")) need the channels.
            image = rasterize_page(page, dpi=self.config.rasterize_dpi, grayscale=False)
        except Exception as exc:  # noqa: BLE001
            rec.add_flag(
                FlagCode.UNREADABLE_PAGE, Severity.ERROR,
                detail=f"rasterization failed: {type(exc).__name__}: {exc}",
            )
            return rec
        rec.add_step("rasterize", "pymupdf", dpi=self.config.rasterize_dpi)
        self._run_image_and_store(image, rec, path, page_index)
        return rec

    # -- plain image file handling --------------------------------------------

    def _process_image_file(self, path: Path) -> PageRecord:
        rec = PageRecord(
            provenance=Provenance(
                path, 0, SourcePath.IMAGE, "plain image file"
            )
        )
        try:
            image = load_image(path)
        except Exception as exc:  # noqa: BLE001
            rec.add_flag(
                FlagCode.UNREADABLE_PAGE, Severity.ERROR,
                detail=f"{type(exc).__name__}: {exc}",
            )
            return rec
        self._run_image_and_store(image, rec, path, 0)
        return rec

    # -- shared image path ----------------------------------------------------

    def _run_image_and_store(
        self, image, rec: PageRecord, path: Path, page_index: int
    ) -> None:
        cleaned = run_image_steps(image, self.config.image_steps, rec)
        rec.image = cleaned
        # rec.gray was populated by run_image_steps (pre-binarization grayscale).
        if self.config.write_images_to is not None:
            stem = f"{path.stem}_p{page_index:04d}"
            try:
                rec.image_path = write_image(
                    cleaned, self.config.write_images_to / f"{stem}.png"
                )
                # Preserve the lossless grayscale next to the binary output.
                if rec.gray is not None:
                    rec.gray_path = write_image(
                        rec.gray, self.config.write_images_to / f"{stem}_gray.png"
                    )
            except Exception as exc:  # noqa: BLE001
                rec.add_flag(
                    FlagCode.PROCESSING_ERROR, Severity.ERROR,
                    detail=f"image write failed: {type(exc).__name__}: {exc}",
                    step="write_image",
                )
