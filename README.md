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

## Degraded pages: preserving faint text

Binarization is irreversible: every stroke that falls below the threshold —
exactly the faded text sitting on a dark stain — is gone. Those faint strokes
are often real Ottoman words or Persian izafet links, not noise, so OttoParse
does **not** treat the binary image as the only output.

- **Grayscale is preserved.** Each image `PageRecord` carries both `image` (the
  binary convenience view) and `gray` (the cleaned, pre-binarization grayscale,
  with all tonal information intact). When writing to disk, a companion
  `{stem}_gray.png` is saved next to the binary `{stem}.png`. A downstream
  OCR/HTR engine or lexicon/language-model stage can mine the grayscale to
  recover glyphs the binary view dropped.
- **Low-confidence regions are flagged, not deleted.** The opt-in `ConfidenceMap`
  step marks faint/stained areas as `Region`s (with a 0-1 confidence score and a
  `LOW_CONFIDENCE_REGION` flag) without altering a single pixel, so later stages
  know where to look and can reconstruct contextually.

```python
from ottoparse import Pipeline
from ottoparse.imaging import (
    HoughDeskew, NlMeansDenoiser, SauvolaBinarizer, ConfidenceMap,
)

pipe = Pipeline(image_steps=[
    HoughDeskew(),
    NlMeansDenoiser(),
    ConfidenceMap(),        # flag faint/stained regions on grayscale; deletes nothing
    SauvolaBinarizer(),
], write_images_to="out/")  # also writes out/<name>_gray.png per page

rec = pipe.process("samples/1-1.jpg")[0]
rec.gray          # lossless grayscale (ndarray)
rec.regions       # low-confidence bounding boxes with confidence scores
```

Place `ConfidenceMap` **before** binarization (so it sees grayscale tones);
after binarization every pixel is pure black/white and no area looks faint.

`ConfidenceMap`'s threshold is adaptive by default (a low percentile of the
page's own tile contrasts), so it flags areas faint *relative to that page*
rather than using a fixed cutoff that misreads uniformly-yellowed paper. Honest
limit: where a stain fully covered the ink there is no signal to recover — the
grayscale and flags preserve what is faint-but-present, not what is gone.

## Artifact removal (periodicals)

Historical periodicals carry non-body material — seals, stamps, editorial
notes, a central title emblem/watermark, and ink blots — that survives
binarization and pollutes line serialization. Two **opt-in** image steps address
this (both off by default, since they are layout-specific and lossy):

- **`ComponentFilter`** — runs after binarization and removes connected
  components whose *shape* looks like an artifact (tiny ghost specks; large,
  densely-filled blobs such as stamps and ink blots) while leaving text alone.
  Robust to where the artifact sits on the page.
- **`RegionMask`** — blanks configurable page regions given as *fractions* of
  page size (e.g. a masthead band and side margins). Use this for the central
  watermark and editorial stamps, which are not distinguishable from text by
  shape. It records the masked bounds and raises a flag, so the (lossy) removal
  is auditable — a masthead holds real metadata you may want to keep.

```python
from ottoparse import Pipeline
from ottoparse.imaging import (
    HoughDeskew, NlMeansDenoiser, SauvolaBinarizer, ComponentFilter, RegionMask,
)

# Clean a periodical front page: strip specks/blobs page-wide, then mask the
# top masthead band that holds the seal, watermark, and editorial stamp.
pipe = Pipeline(image_steps=[
    HoughDeskew(),
    NlMeansDenoiser(),
    SauvolaBinarizer(),
    ComponentFilter(),       # remove specks and solid blobs (stamps, blots)
    RegionMask(top=0.22),    # suppress the masthead band (tune per layout)
])
records = pipe.process("samples/1-1.jpg")
```

Order matters: `ComponentFilter` must come **after** binarization (it needs a
binary image), and `RegionMask` is typically last. The `top` fraction is
layout-specific — tune it on your own scans.

## Manual testing

A small command-line runner, `run_ottoparse.py`, lets you try the pipeline on
your own files without writing any code. Drop PDFs or images into `samples/`
(both `samples/` and the `results/` output folder are git-ignored, so private
archival material is never committed) and run:

```bash
# a single file
.venv/bin/python run_ottoparse.py samples/your_doc.pdf

# a whole folder
.venv/bin/python run_ottoparse.py samples/
```

For each page it prints the routing decision, the steps that ran, any quality
flags (with measured values), a text preview for text-layer pages, and the path
to the cleaned image for scanned pages. Cleaned images are written to
`results/` by default.

Supported inputs: PDF, and images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`,
`.bmp`, `.webp`).

### Options

| Flag | Description |
|------|-------------|
| `--any-script` | Accept text layers of any script. Use this for Latin / non-Arabic test PDFs; otherwise their text pages are flagged *suspect* and routed to the image path. Leave off for real Ottoman documents. |
| `--out DIR` | Directory for cleaned images (default: `results/`). |
| `--dpi N` | Rasterization DPI for image-only PDF pages (default: 300). |
| `--json` | Also print each page record as JSON (the full Phase 2 hand-off object). |

```bash
# e.g. test a Latin PDF, higher DPI, custom output, with JSON dump
.venv/bin/python run_ottoparse.py samples/doc.pdf --any-script --dpi 400 --out results/ --json
```

Note: the default router is tuned for Perso-Arabic text layers. A real Ottoman
PDF with an Arabic text layer keeps its text when run without `--any-script`; a
Latin-script test PDF needs `--any-script` or its text pages will be rasterized.

## Running the tests

```bash
.venv/bin/pytest          # run the full suite
.venv/bin/ruff check .    # lint
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full, step-by-step history of changes.

## License

Library code is MIT. Note that the optional runtime dependency **PyMuPDF is
AGPL-3.0**; it is used here for research purposes.
