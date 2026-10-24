# Changelog

All notable changes to OttoParse are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Focus: stained pages. The 0.2.0 default chain (CLAHE + Sauvola) turned stain
outlines and paper folds into thick black contours running through the text.

### Added

- Background normalization (`ottoparse/imaging/background.py`),
  `BackgroundNormalizer` — estimates the paper tone with a grayscale
  morphological closing (disk wider than any stroke) plus Gaussian blur, and
  divides the page by it. Stains, foxing, and uneven lighting flatten to white
  while ink keeps its contrast, including inside a stain.
  - `channel`: `"luminance"` (default), `"red"`, `"green"`, or `"blue"`.
  - `kernel_size`: automatic by default (~2% of the page's shorter side,
    ~41 px at 300 DPI); `blur_sigma` defaults to `kernel_size / 5`.
  - Logs the channel actually used (`"gray"` for single-channel input) and the
    resolved kernel to the processing history.
- `tests/test_background.py` — synthetic stain flattening, stain-outline
  suppression vs. raw Sauvola, channel selection, auto kernel, validation.
- Projection-profile deskew (`ottoparse/imaging/deskew.py`), `ProjectionDeskew`
  — searches +/-2x `max_angle` (coarse 0.5°, then fine 0.05°) for the rotation
  that maximizes the sharpness (sum of squared row-to-row differences) of the
  horizontal ink profile. It estimates on a ~1200 px working copy, flattened
  with `BackgroundNormalizer`'s background estimate before Otsu, and keeps only
  glyph-sized connected components. That way yellowed paper, stains, dark scan
  margins and column rules don't steer it. Takes ~30 ms per page, logs
  strategy `"projection"`, and keeps `HoughDeskew`'s `max_angle` /
  `min_correction` semantics and flags.
- A `periodical_page` test fixture: a rotated, real-like RGB page (yellowed
  paper, stain, ragged dark margin, two columns of broken lines imitating Ottoman Turkish print
  with dots, white rotation corners). Tests check that `ProjectionDeskew`
  recovers the rotation and levels the page. A regression test on
  `samples/1-*.jpg` runs when the private samples are present and skips
  otherwise.

### Fixed

- Deskew never ran on real input. `HoughDeskew` flagged `SKEW_UNDETECTED`
  ("no near-horizontal lines found") on every real periodical sample.
  `HoughLinesP` needs one unbroken edge segment of at least a quarter of the
  page width (~500 px). On two-column Ottoman Turkish (Perso-Arabic script)
  print, word gaps are wider than the 15 px dilation plus the 20 px
  `maxLineGap`, so no line ever gets that long. The synthetic `cv2.putText` pages have continuous Latin baselines,
  which hid this. The default chain now uses `ProjectionDeskew`, which
  estimates -0.40°, -0.85° and -0.10° on `samples/1-1`, `1-2` and `1-4` and
  recovers 3°/-5°/8° rotations of them to within ~0.4°. `HoughDeskew` is
  unchanged and still available.

### Changed

- **Default image chain** is now deskew → `BackgroundNormalizer` → denoise →
  `ConfidenceMap` → `OtsuBinarizer` → `ComponentFilter`. CLAHE is removed from
  the default (still available as `ClaheEnhancer`), and Sauvola is replaced by
  Otsu: once the background is flat a global threshold suffices and kept
  strokes and dots more intact than Sauvola on the samples. On the three
  stained periodical samples the stain contours and fold lines disappeared and
  the ink fraction fell from ~7.6-8.6% to ~5.9-6.9% (the removed pixels were
  stain edges, not text).
- PDF image-only pages are rasterized in **RGB** instead of grayscale, so
  color-aware steps can use the channels. Steps that need grayscale still
  convert internally.
- `record.gray` with the default chain is now the background-normalized
  grayscale.
- Because the page is normalized before denoising, `NlMeansDenoiser`'s
  quality measurements now see flat paper; `BLEED_THROUGH` no longer fires on
  merely yellowed or stained paper.

- **Foxing suppression in `OtsuBinarizer`** (on by default). On a flattened
  page Otsu's threshold sits high (~190-197 on the samples, because faded ink
  is only mid-gray), so faint foxing spots also fall below it. They came out
  as solid black dots, and next to a letter they read as a nokta. Printed ink,
  even faded, has a dark core (darkest pixel ~90-100 on the samples) and
  foxing does not (~160). Otsu still defines the shapes, but a component is
  kept only if some pixel is darker than
  `ink_level + core_contrast * (otsu - ink_level)` (`core_contrast` 0.4).
  - `ink_level` is **local**: the 10th percentile of ink in each `window`
    (150 px) tile, interpolated across the page. A page-wide level deleted
    whole faded letters under stains (e.g. "دى" on 1-1). Tiles with under 2%
    ink fall back to the page-wide level, so a tile holding only foxing can't
    become its own reference.
  - Two shapes are always kept: upright strokes (`upright_aspect` 2.5, e.g.
    elifs) and horizontal ones (`rule_aspect` 5, printed rules that layout
    analysis needs).
  - `core_contrast=None` restores plain Otsu. Logs strategy
    `"otsu_hysteresis"` with a `removed_faint` count: 347-504 marks per sample
    page (foxing, stain blobs, pencil annotations, specks). The foxing dot
    above "سماوى" on 1-1 is removed; the stain-faded letters, rules and the
    dots checked near ش are kept.

### Known limitations

- Faint marks *touching* a glyph merge into its component and are kept by the
  foxing filter. The filter also removes faint print that has no dark core,
  e.g. very light punctuation dots.
- Solid dark areas wider than the normalization kernel (large calligraphic
  mastheads, filled illustrations) come out hollow; raise `kernel_size` if
  that matters.
- The darkest core of a heavy stain can still leave a small blob after
  thresholding.

## [0.2.0] — 2026-10-17

Focus of this release: handling degraded, stained late-Ottoman periodical
scans without destroying faint text, and enriching the default cleanup chain.

### Added

- CLAHE contrast enhancement (`ottoparse/imaging/enhance.py`), `ClaheEnhancer`
  — wraps OpenCV's CLAHE to equalize contrast in local tiles on the grayscale,
  lifting faint text on degraded/stained historical print before denoising and
  binarization. Tunable `clip_limit` (default 2.0) and `tile_grid_size`
  (default 8x8). Validated on a real stained periodical scan: at `clip_limit=2.0`
  lines lost under a blot became legible, while higher limits traded legibility
  for amplified noise.
- Grayscale preservation alongside binarization (non-destructive output). The
  cleaned, pre-binarization grayscale is now kept so faint strokes that fall
  below the binary threshold — the text under a stain — are never lost:
  - `PageRecord` gained `gray` (the lossless grayscale) and `gray_path`
    fields; `image` remains the binary convenience view. `run_image_steps`
    snapshots the grayscale feeding the first binarizer (or the final cleaned
    image if no binarizer runs), and the pipeline writes a companion
    `{stem}_gray.png` when `write_images_to` is set.
- Confidence mapping guardrail (`ottoparse/imaging/confidence.py`),
  `ConfidenceMap` — a **non-destructive** step (now part of the default chain,
  before binarization) that flags faint/stained areas instead of deleting them,
  so a downstream lexicon matcher or language model can reconstruct ambiguous
  glyphs from the preserved grayscale:
  - Divides the page into tiles, measures local contrast, and flags low-contrast
    (but non-blank) tiles, merging adjacent ones into bounding-box regions.
  - Threshold is **adaptive by default** (a low percentile of the page's own
    tile contrasts), since a fixed cutoff cannot tell a crisp page from a
    uniformly-faded one; an absolute `min_contrast` override is available.
  - Records each area as a `Region` (new `PageRecord.regions` list + `Region`
    model type) with a 0-1 confidence score, plus a page-level
    `LOW_CONFIDENCE_REGION` flag (new code). Returns pixels untouched.
- Artifact removal for periodicals (`ottoparse/imaging/artifacts.py`):
  - `ComponentFilter` — post-binarization connected-component removal of ghost
    specks and large solid blobs (stamps, seals, ink blots) by shape, leaving
    text intact. Defaults (`min_area=6`, `max_area=900`, `blob_fill=0.45`) were
    tuned against a real periodical scan. Now part of the default chain (runs
    last) to clean the speckle CLAHE/denoise leave behind.
  - `RegionMask` (opt-in) — blanks configurable page regions given as fractions
    of page
    size (top/bottom/left/right); records the bounds and raises a
    `REGION_MASKED` warning so the lossy masking is auditable. Reliable for the
    central masthead watermark and editorial stamps, which shape-based filtering
    cannot distinguish from text.
  - New flag codes `ARTIFACT_REMOVED` and `REGION_MASKED`.
  - Documented in the README with a periodical front-page example, and covered
    by `tests/test_artifacts.py`.
- `run_ottoparse.py`: a command-line runner for manual testing. Point it at a
  single file or a folder of PDFs/images and it prints a per-page report
  (routing decision, steps run, quality flags with measures, text preview, and
  cleaned-image path), writing cleaned images to an output directory. Options:
  `--out`, `--dpi`, `--any-script` (accept text layers of any script), and
  `--json` (dump each record as JSON).
- `samples/` folder (with a short README) as a drop spot for test documents.
- "Manual testing" and "Running the tests" sections in the README documenting
  the runner, its options, and the test/lint commands.

### Changed

- Default image chain expanded from **deskew → denoise → Sauvola** to
  **deskew → CLAHE → denoise → ConfidenceMap → Sauvola → ComponentFilter**.
  This changes the output of `Pipeline()` for all pages, including clean ones,
  and adds per-page work (ConfidenceMap tiles the whole page). To restore a
  leaner chain, pass an explicit `image_steps`.
- `.gitignore`: ignore the `samples/` and `results/` folders so private
  archival material and generated outputs are never committed.

## [0.1.0] — 2026-10-17

First release. Implements **Phase 1 — Intake & Pre-Processing** for late-Ottoman
printed (*Matbu*) documents. Scope is Phase 1 only: Phase 2 (OCR/HTR engines) and
Phase 3 (lexicon, tri-layer normalization, corpus packaging) are intentionally
out of scope, with the `PageRecord` output designed as a stable hand-off contract
for them.

Below is the step-by-step build history, in the order each piece was added.

### Step 1 — Project scaffolding

- Created `pyproject.toml` (hatchling build backend), targeting **Python ≥3.13**
  after verifying Python 3.13.9 was available locally and has prebuilt wheels for
  every native dependency (3.15 was the newest release but risked missing wheels).
- Declared dependencies: `numpy`, `opencv-python-headless` (headless chosen over
  `opencv-python` to avoid GUI/Qt deps in a library), `scikit-image`, `Pillow`,
  `PyMuPDF`. Dev extras: `pytest`, `ruff`.
- Added `LICENSE` (MIT). Noted that PyMuPDF is AGPL-3.0 but acceptable for
  research use.
- Wrote the initial `README.md` (scope, design principles, install, usage).
- Created the package skeleton: `ottoparse/{core,model,intake,imaging}/`.
- Created a virtual environment on Python 3.13.9 and installed in editable mode;
  all dependencies resolved to prebuilt arm64 wheels with no compilation.

### Step 2 — Data model (the Phase 1 → Phase 2 hand-off contract)

- `ottoparse/model/flags.py`:
  - `Severity` enum (`INFO`, `WARNING`, `ERROR`).
  - `FlagCode` enum of stable, machine-readable issue codes (routing, deskew,
    denoise, binarization, and IO categories).
  - `IssueFlag` dataclass: `code`, `severity`, optional numeric `measure`,
    `detail`, and originating `step`, with `to_dict()`.
- `ottoparse/model/record.py`:
  - `SourcePath` enum (`TEXT_LAYER`, `IMAGE`).
  - `Provenance` (source file, page index, path taken, human-readable reason),
    `ProcessingStep` (name, strategy, params), and `PageRecord`.
  - `PageRecord` exposes `add_flag()` / `add_step()` helpers, `is_text` /
    `is_image` / `has_errors` / `max_severity` properties, and a JSON-friendly
    `to_dict()`. A page is **either** text or image, never both; failures become
    flags, never dropped pages.
- Verified records build and serialize cleanly to JSON.

### Step 3 — Core contracts (swappable strategies)

- `ottoparse/core/strategy.py`: `@runtime_checkable` protocols `ImageStep`
  (`name` + `apply(image, record) -> ndarray`) and `PageRouter`
  (`decide(page) -> RouteDecision`), plus the `RouteDecision` dataclass.
  Structural typing means any object with the right shape plugs in — no base
  class required.
- `ottoparse/core/config.py`: `PipelineConfig` bundling the router, the ordered
  `image_steps`, `rasterize_dpi`, and optional `write_images_to`; validates DPI
  and normalizes the output path.
- Verified a duck-typed step satisfies the `ImageStep` protocol and that config
  validation rejects a non-positive DPI.

### Step 4 — PDF path (routing, extraction, rasterization)

- `ottoparse/intake/script.py`: Perso-Arabic Unicode range check and
  `arabic_ratio()` (fraction of letters that are Arabic-script; ignores
  non-letters). Used to tell a genuine Ottoman text layer from mojibake.
- `ottoparse/intake/pdf_text.py`: `extract_page_text()` (PyMuPDF) and a
  conservative, deterministic `normalize_text()` — Unicode NFC, strip control
  chars, rejoin Latin words hyphenated across line breaks, collapse whitespace,
  tidy blank lines. **No transliteration** (that belongs to Phase 3).
- `ottoparse/intake/router.py`: `DefaultRouter` makes the text-vs-image decision
  **per page**: too little text → image; enough text but low Arabic ratio →
  flagged suspect and routed to image; otherwise → text layer. Every decision
  carries an auditable reason, and the router is swappable.
- `ottoparse/intake/rasterize.py`: `rasterize_page()` renders image-only PDF
  pages to grayscale `uint8` arrays at a configurable DPI.
- Switched the PyMuPDF import from the deprecated `import fitz` to
  `import pymupdf`.
- Verified end-to-end on a synthetic 2-page PDF (text page vs image-only page).

### Step 5 — Image path (deskew, denoise, binarize)

- `ottoparse/imaging/utils.py`: `to_grayscale()` (handles grayscale / RGB /
  RGBA) and `is_binary()`.
- `ottoparse/imaging/deskew.py`: `HoughDeskew` estimates skew via the
  probabilistic Hough transform and rotates the page level; flags `SEVERE_SKEW`
  / `SKEW_UNDETECTED`.
- `ottoparse/imaging/denoise.py`: `NlMeansDenoiser` (edge-preserving Non-Local
  Means) plus cheap detectors that flag `HIGH_NOISE`, `LOW_CONTRAST`, and
  `BLEED_THROUGH` with numeric measures.
- `ottoparse/imaging/binarize.py`: `SauvolaBinarizer` (default — local adaptive,
  best for degraded scans), `OtsuBinarizer`, and `AdaptiveGaussianBinarizer`.
  All output foreground = 0 / background = 255 and flag
  `BINARIZATION_DEGENERATE` on near-single-color results.
- `ottoparse/imaging/pipeline.py`: `run_image_steps()` chains steps, threading
  one record through; a step that raises is caught as a `PROCESSING_ERROR` flag
  so a single step failure never aborts the page.
- **Bug fix (OpenCV 5.0 compatibility):** OpenCV 5.0 changed `HoughLinesP` to
  return shape `(N, 4)` instead of `(N, 1, 4)`. Deskew now reshapes to `(N, 4)`
  so it works on both OpenCV 4.x and 5.x.
- **Bug fix (noise-robust deskew):** angle estimation on raw noisy scans
  returned 0°. The estimator now Otsu-binarizes and dilates a working copy
  before edge/line detection, so it locks onto text baselines rather than
  speckle. Verified across +5°, −3°, 0°, +8°.
- **Refactor:** `FlagCode` changed from `(str, Enum)` to `enum.StrEnum`
  (idiomatic on 3.13; serializes straight to JSON).

### Step 6 — Top-level orchestrator

- `ottoparse/io.py`: `load_image()` / `write_image()` with correct BGR↔RGB
  handling, plus the supported `IMAGE_EXTENSIONS` / `PDF_EXTENSIONS` sets.
- `ottoparse/pipeline.py`: `Pipeline` — the single public entry point.
  `process(source) -> list[PageRecord]` dispatches on file type, fans a document
  out into one record per page, routes each PDF page, runs the matching path, and
  optionally writes cleaned images to disk (`{stem}_p{index:04d}.png`).
  Unreadable pages and rasterization/write failures become error flags rather
  than crashes. `default_image_steps()` wires deskew → denoise → Sauvola.
- `ottoparse/__init__.py`: exports the public API (`Pipeline`, `PageRecord`,
  `FlagCode`, `Severity`, `SourcePath`, etc.).
- Verified end-to-end: mixed PDF (text page vs rasterized-and-cleaned image
  page), plain PNG scan (4° skew corrected), disk output, and JSON serialization.

### Step 7 — Test suite

- Added `tests/` with shared fixtures (`conftest.py`) that synthesize images and
  PDFs in a temp dir, plus `test_model.py`, `test_intake.py`, `test_imaging.py`,
  and `test_pipeline.py`.
- **31 tests, all passing** (~2.2s). `ruff` reports no issues across the package
  and tests.

[Unreleased]: https://github.com/your-group/OttoParse/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/your-group/OttoParse/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/your-group/OttoParse/releases/tag/v0.1.0
