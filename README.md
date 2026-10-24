# OttoParse

Phase 1 intake & pre-processing for late-Ottoman **printed (Matbu)** documents
in Ottoman Turkish (Perso-Arabic script).

OttoParse ingests degraded archival scans and encoded PDFs and produces clean,
per-page records ready to hand to a downstream OCR/HTR stage (Phase 2). It does
**not** perform OCR itself — its job is to route, clean, and normalize input, and
to flag quality problems rather than silently drop them.

## Scope (v0.3)

This release implements **Phase 1 only**:

- **1.1 Document routing** — per *page*, decide whether a PDF page already has a
  usable text layer (extract it) or is image-only (rasterize and clean it).
- **PDF text path** — deterministic extraction via PyMuPDF + regex normalization.
- **1.2 Deskewing & alignment** — correct page rotation. `ProjectionDeskew`
  (default) picks the angle that makes the row ink profile sharpest, which
  works on Ottoman Turkish (Perso-Arabic script) print; `HoughDeskew` (long
  baselines) remains available for Latin print.
- **Background normalization** — divide the page by its estimated paper tone
  so stains, foxing, and uneven lighting flatten to white while ink keeps its
  contrast. (CLAHE remains available as an opt-in alternative.)
- **1.3 Denoising** — remove speckle and background noise.
- **Confidence mapping** — flag faint/stained regions (non-destructive) so
  downstream stages can reconstruct ambiguous glyphs; nothing is deleted.
- **1.4 Binarization** — Otsu by default (a global threshold is enough once
  the background is flat). Otsu drops faint stand-alone marks with no dark ink
  core, such as foxing spots that would otherwise read as dots
  (`core_contrast=None` turns this off). Sauvola and adaptive-Gaussian remain
  available as alternatives, for pages that are not background-normalized or
  as comparison baselines.
- **Artifact removal** — `ComponentFilter` (default) strips ghost specks and
  solid blobs (stamps, seals, ink blots) after binarization; `RegionMask`
  (opt-in) blanks masthead bands.

The default image chain (`ottoparse.default_image_steps()`) is
**`ProjectionDeskew` → `BackgroundNormalizer` → `NlMeansDenoiser` →
`ConfidenceMap` → `OtsuBinarizer` → `ComponentFilter`**.
The cleaned grayscale is preserved alongside the binary image so faint text is
never lost. Every step is swappable; pass an explicit `image_steps` to customize.

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

# Customized pipeline (this mirrors the default chain)
from ottoparse.imaging import (
    ProjectionDeskew, BackgroundNormalizer, NlMeansDenoiser,
    ConfidenceMap, OtsuBinarizer, ComponentFilter,
)
from ottoparse.intake import DefaultRouter

pipe = Pipeline(
    router=DefaultRouter(min_chars=50),
    image_steps=[
        ProjectionDeskew(),
        BackgroundNormalizer(),        # flatten stains / uneven paper
        NlMeansDenoiser(),
        ConfidenceMap(),               # flag faint regions (on grayscale)
        OtsuBinarizer(),
        ComponentFilter(),             # remove specks/blobs (on binary)
    ],
    write_images_to="out/",  # optional disk output
)
records = pipe.process("archive_scan.pdf")
```

## Background normalization (stains & uneven paper)

A stain darkens the paper *multiplicatively* — the paper and the ink under a
blot are dimmed by about the same factor. A local thresholder like Sauvola sees
the edge between clean and stained paper as a sharp brightness step and marks
it as ink, so stain outlines come out as thick black contours through the text.

`BackgroundNormalizer` undoes the darkening instead: it estimates the paper tone
under every pixel (a grayscale morphological closing with a disk wider than any
text stroke, then a Gaussian blur) and divides the page by it. Clean and stained
paper both become near-white; ink keeps its contrast relative to its *own*
background, including inside a stain. After this, a global Otsu threshold is
enough. On the stained periodical samples this removed the stain contours and
fold lines that the previous CLAHE + Sauvola chain produced.

```python
from ottoparse.imaging import BackgroundNormalizer

BackgroundNormalizer()                   # luminance, auto kernel (~2% of page side)
BackgroundNormalizer(channel="blue")     # use the blue channel of a color scan
BackgroundNormalizer(kernel_size=41)     # fixed kernel (odd, wider than strokes)
```

- **`channel`** — `"luminance"` (default), `"red"`, `"green"`, or `"blue"`.
  Yellow/brown stains are closer to paper in blue, but on the samples tested
  luminance gave equal or cleaner results (the blue channel of a JPEG is
  noisier). Image files and PDF pages are now kept in color until this step, so
  either choice works; grayscale input uses its single channel.
- **`kernel_size`** — must exceed the widest text stroke. The automatic value
  (~41 px on a 300 DPI page) suits periodical body text.
- **Limitation:** solid dark areas *wider* than the kernel (large calligraphic
  mastheads, filled illustrations, thick bars) are treated as background and
  come out hollow or blank. Use a larger `kernel_size` if that matters.

## Deskew on real scans

`HoughDeskew` looks for long, straight baselines. Ottoman Turkish print has
none long enough: word gaps break every line and periodicals are set in
narrow columns, so on real scans it found no lines and the page was never
deskewed. `ProjectionDeskew` instead rotates a downscaled, background-flattened
copy over candidate angles and keeps the one where ink piles up into the
sharpest horizontal bands (a projection profile). It keeps only glyph-sized
components, so stains, scan margins and column rules do not steer it.

```python
from ottoparse.imaging import ProjectionDeskew

ProjectionDeskew()                       # +/-15 deg, ignore < 0.1 deg
ProjectionDeskew(max_angle=10, min_correction=0.2)
```

It records the angle and whether it was applied in the processing history,
and flags `SEVERE_SKEW` / `SKEW_UNDETECTED` the same way `HoughDeskew` does.

## Foxing and faint marks (binarization)

After background normalization the paper is near-white but faded ink is only
mid-gray, so Otsu's threshold sits high and faint foxing spots fall below it
too. They come out as solid dots, and next to a letter a dot can change it
(ر/ز, ح/خ, ب/ت). `OtsuBinarizer` therefore uses two levels (hysteresis):
Otsu defines each mark's shape, but a mark is kept only if some pixel is
darker than a stricter *core* level. Printed ink, even faded, has a dark core;
foxing does not.

- **The core level is local.** It is measured from the ink in each ~150 px
  tile, so faded letters under a stain are judged against their own faded
  neighbours, not against clean print elsewhere. Tiles with little ink use
  the page-wide level, so foxing alone in a margin cannot vouch for itself.
- **Thin shapes are always kept:** tall, narrow strokes (an elif) and long
  horizontal ones (printed rules, which layout analysis needs).
- **Only stand-alone marks are removed.** A faint mark touching a letter
  merges with it and stays.

```python
from ottoparse.imaging import OtsuBinarizer

OtsuBinarizer()                          # defaults: core_contrast=0.4, window=150
OtsuBinarizer(core_contrast=0.3)         # keep more faint marks
OtsuBinarizer(core_contrast=None)        # plain Otsu, no foxing filter
```

The step logs `removed_faint` (the number of marks removed) in the processing
history. If real nokta, hemze or medde go missing on a corpus, lower
`core_contrast`; the preserved grayscale (`record.gray`) still has every mark.

## Alternative binarizers

`OtsuBinarizer` is the default because it works once `BackgroundNormalizer`
has flattened the page. Two local thresholders stay available:

- **`SauvolaBinarizer`** (`window_size=25`, `k=0.2`) — a per-pixel threshold
  from the local mean and spread. Use it for pages you do not
  background-normalize. On stained pages its small window treats stain edges
  as ink.
- **`AdaptiveGaussianBinarizer`** (`block_size=31`, `c=10`) — OpenCV's
  adaptive Gaussian threshold; a lighter local method.

Neither has the foxing filter, so they are also useful as baselines when
comparing binarization methods:

```python
from ottoparse import Pipeline
from ottoparse.imaging import (
    ProjectionDeskew, NlMeansDenoiser, ConfidenceMap, SauvolaBinarizer, ComponentFilter,
)

# The pre-0.3.0 style chain: no background normalization, Sauvola instead of Otsu.
baseline = Pipeline(image_steps=[
    ProjectionDeskew(), NlMeansDenoiser(), ConfidenceMap(),
    SauvolaBinarizer(), ComponentFilter(),
])
```

## Contrast enhancement (CLAHE, opt-in)

Degraded historical paper has wildly uneven text-to-background contrast: crisp
in clean areas, almost none where a stain or foxing darkens the page. A single
global stretch cannot fix that. `ClaheEnhancer` applies OpenCV's CLAHE (Contrast
Limited Adaptive Histogram Equalization) to the grayscale, equalizing contrast
in local tiles so each region — including a stained patch — gets its own stretch
and faint ink is pulled apart from its background.

It runs **on the grayscale, before denoise and binarization**. It is no longer
in the default chain: on stained pages it amplifies stain edges, which Sauvola
then turns into black contours — `BackgroundNormalizer` handles that case
better. Use it as an alternative, e.g. on evenly lit but faded scans. The clip limit caps local
amplification: higher values recover more faint text but amplify more speckle,
foxing, and bleed-through, so raise it only as far as a given corpus tolerates.

```python
from ottoparse.imaging import ClaheEnhancer

ClaheEnhancer(clip_limit=2.0, tile_grid_size=(8, 8))  # defaults
```

To use it, pass an explicit `image_steps` list with `ClaheEnhancer` in place of
`BackgroundNormalizer`.

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
  recover glyphs the binary view dropped. With the default chain this is the
  *background-normalized* grayscale (stains flattened, ink contrast kept),
  which many HTR engines accept directly.
- **Low-confidence regions are flagged, not deleted.** The `ConfidenceMap`
  step marks faint/stained areas as `Region`s (with a 0-1 confidence score and a
  `LOW_CONFIDENCE_REGION` flag) without altering a single pixel, so later stages
  know where to look and can reconstruct contextually.

Both are part of the default chain:

```python
from ottoparse import Pipeline

pipe = Pipeline(write_images_to="out/")  # also writes out/<name>_gray.png per page

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
binarization and pollutes line serialization. Two image steps address this:

- **`ComponentFilter`** (in the default chain) — runs after binarization and
  removes connected components whose *shape* looks like an artifact (tiny
  ghost specks; large, densely-filled blobs such as stamps and ink blots) while
  leaving text alone. Robust to where the artifact sits on the page.
- **`RegionMask`** (opt-in, since it is layout-specific and lossy) — blanks
  configurable page regions given as *fractions* of page size (e.g. a masthead
  band and side margins). Use this for the central watermark and editorial
  stamps, which are not distinguishable from text by shape. It records the
  masked bounds and raises a flag, so the (lossy) removal is auditable — a
  masthead holds real metadata you may want to keep.

```python
from ottoparse import Pipeline, default_image_steps
from ottoparse.imaging import RegionMask

# Clean a periodical front page: the default chain (which already ends with
# ComponentFilter), then mask the top masthead band that holds the seal,
# watermark, and editorial stamp.
pipe = Pipeline(image_steps=[
    *default_image_steps(),
    RegionMask(top=0.22),    # suppress the masthead band (tune per layout)
])
records = pipe.process("samples/1-1.jpg")
```

Order matters: `ComponentFilter` must come **after** binarization (it needs a
binary image), and `RegionMask` is typically last. The `top` fraction is
layout-specific — tune it on your own scans.

## Project layout

```text
ottoparse/
  pipeline.py        Pipeline orchestrator and default_image_steps()
  io.py              image loading/writing, supported extensions
  core/              ImageStep / PageRouter protocols, PipelineConfig
  model/             PageRecord, Provenance, ProcessingStep, Region, IssueFlag
  intake/            per-page routing, PDF text extraction, rasterization,
                     Perso-Arabic script heuristic
  imaging/           deskew, background normalization, CLAHE, denoise,
                     confidence map, binarizers, artifact removal
tests/               pytest suite (synthetic inputs; sample tests skip
                     when samples/ is absent)
docs/                design notes (e.g. tri-level thresholding plan review)
run_ottoparse.py     command-line runner for manual testing
```

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
| `--any-script` | Accept text layers of any script. Use this for Latin / non-Perso-Arabic test PDFs; otherwise their text pages are flagged *suspect* and routed to the image path. Leave off for real Ottoman documents. |
| `--out DIR` | Directory for cleaned images (default: `results/`). |
| `--dpi N` | Rasterization DPI for image-only PDF pages (default: 300). |
| `--json` | Also print each page record as JSON (the full Phase 2 hand-off object). |

```bash
# e.g. test a Latin PDF, higher DPI, custom output, with JSON dump
.venv/bin/python run_ottoparse.py samples/doc.pdf --any-script --dpi 400 --out results/ --json
```

Note: the default router is tuned for Perso-Arabic text layers. A real Ottoman
PDF with a Perso-Arabic text layer keeps its text when run without
`--any-script`; a Latin-script test PDF needs `--any-script` or its text pages
will be rasterized.

## Running the tests

```bash
.venv/bin/pytest          # run the full suite
.venv/bin/ruff check .    # lint
```

The suite builds its inputs synthetically. A few regression tests run on the
private scans `samples/1-*.jpg` when they are present and are skipped
otherwise.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full, step-by-step history of changes.

## License

Library code is MIT. Note that the optional runtime dependency **PyMuPDF is
AGPL-3.0**; it is used here for research purposes.
