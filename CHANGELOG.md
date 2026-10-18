# Changelog

All notable changes to OttoParse are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  `ConfidenceMap` — a **non-destructive** opt-in step that flags faint/stained
  areas instead of deleting them, so a downstream lexicon matcher or language
  model can reconstruct ambiguous glyphs from the preserved grayscale:
  - Divides the page into tiles, measures local contrast, and flags low-contrast
    (but non-blank) tiles, merging adjacent ones into bounding-box regions.
  - Threshold is **adaptive by default** (a low percentile of the page's own
    tile contrasts), since a fixed cutoff cannot tell a crisp page from a
    uniformly-faded one; an absolute `min_contrast` override is available.
  - Records each area as a `Region` (new `PageRecord.regions` list + `Region`
    model type) with a 0-1 confidence score, plus a page-level
    `LOW_CONFIDENCE_REGION` flag (new code). Returns pixels untouched.
- Artifact removal for periodicals (`ottoparse/imaging/artifacts.py`), two
  opt-in image steps (off by default):
  - `ComponentFilter` — post-binarization connected-component removal of ghost
    specks and large solid blobs (stamps, seals, ink blots) by shape, leaving
    text intact. Defaults (`min_area=6`, `max_area=900`, `blob_fill=0.45`) were
    tuned against a real periodical scan.
  - `RegionMask` — blanks configurable page regions given as fractions of page
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

- Default image chain is now **deskew → CLAHE → denoise → Sauvola** (CLAHE
  added at `clip_limit=2.0`). This changes the output of `Pipeline()` for all
  pages, including clean ones; to opt out, pass an explicit `image_steps`
  without `ClaheEnhancer`.
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

[Unreleased]: https://github.com/your-group/OttoParse/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-group/OttoParse/releases/tag/v0.1.0
