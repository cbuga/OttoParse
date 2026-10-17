# OttoParse

Phase 1 intake & pre-processing for late-Ottoman **printed (Matbu)** documents.

OttoParse ingests degraded archival scans and encoded PDFs and produces clean,
per-page records ready to hand to a downstream OCR/HTR stage (Phase 2). It does
**not** perform OCR itself — its job is to route, clean, and normalize input, and
to flag quality problems rather than silently drop them.

## Scope (v0.1)

This release implements **Phase 1 only**:

- **1.1 Document routing** — per *page*, decide whether a PDF page already has a
  usable text layer (extract it) or is image-only (rasterize and clean it).
- **PDF text path** — deterministic extraction via PyMuPDF + regex normalization.
- **1.2 Deskewing & alignment** — correct page rotation.
- **1.3 Denoising** — remove speckle and background noise.
- **1.4 Adaptive binarization** — Sauvola by default (best for degraded scans),
  with Otsu and adaptive-Gaussian available as swappable alternatives.

Out of scope for now: Phase 2 (layout analysis, OCR/HTR engines) and Phase 3
(lexicon resolution, tri-layer normalization, corpus packaging). The Phase 1
output (`PageRecord`) is designed as a stable hand-off contract so those phases
can plug in later without reshaping Phase 1.

## Design principles

- **Every step is a swappable strategy.** Routing, deskew, denoise, and binarize
  each sit behind a small protocol, so algorithms (or whole engines later) can be
  replaced without touching the pipeline.
- **Nothing is silently dropped.** Every input page produces exactly one
  `PageRecord`. Problems become `IssueFlag`s with a severity, not crashes.
- **Provenance travels with the data.** Each record records where it came from,
  which path it took and why, and which processing steps ran with what parameters.

## Install

Requires Python 3.13+.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```python
from ottoparse import Pipeline

# Simplest usage: one input -> list[PageRecord]
records = Pipeline().process("archive_scan.pdf")

for rec in records:
    print(rec.provenance.page_index, rec.provenance.source_path, rec.flags)

# Customized pipeline
from ottoparse.imaging import HoughDeskew, NlMeansDenoiser, SauvolaBinarizer
from ottoparse.intake import DefaultRouter

pipe = Pipeline(
    router=DefaultRouter(min_chars=50),
    image_steps=[HoughDeskew(), NlMeansDenoiser(), SauvolaBinarizer(window_size=25)],
    write_images_to="out/",  # optional disk output
)
records = pipe.process("archive_scan.pdf")
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full, step-by-step history of changes.

## License

Library code is MIT. Note that the optional runtime dependency **PyMuPDF is
AGPL-3.0**; it is used here for research purposes.
