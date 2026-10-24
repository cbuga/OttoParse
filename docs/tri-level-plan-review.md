# Review: Tri-Level Semi-Adaptive Thresholding Plan

Reviewed against:

- Shobha Rani et al., *Restoration of deteriorated text sections in ancient
  document images using a tri-level semi-adaptive thresholding technique*,
  Automatika 63:2 (2022), 378–398. doi:10.1080/00051144.2022.2042462
- The OttoParse codebase at v0.2.0 (`develop`).

**Verdict:** the paper's results are much weaker than "excellent", and parts of
its method don't hold together as written. One idea is worth taking: detecting
stains from colour channels. The plan mostly follows the paper, but its last
step would harm Ottoman print. The plan also needs an evaluation harness before
any of it can be shown to beat the current Sauvola chain.

---

## 1. The paper

### Method

1. Linear contrast stretch to the full dynamic range.
2. "Gradient image": `Id = G − (R+B)/2`, then Otsu, giving a reference mask
   `Iref`. Pixels where `Iref = 1` are blacked out in the original, giving
   `Gd`. Despite the name, this is a colour-difference map, not a gradient.
3. Level 1: pixels of the grayscale at or above a mean threshold `T1` become
   255; the rest are kept.
4. Level 2: threshold at the median of the distinct grey levels of the level-1
   output.
5. Level 3: sort the distinct grey levels, split them into four groups, and
   threshold at the mean of group 2.
6. "Pseudo-colouring": Otsu on `Gd`, keep components with area ≥ 500 px as a
   stain mask, then re-threshold inside the mask at `mean(g3) / 2`.
7. Morphological bridging, then dilation.

### Results are modest

- Average F-measure is **65.7 on DIBCO** and **55.2 on palm leaf**. Strong
  DIBCO methods score around 90.
- In the paper's own tables, plain Sauvola beats or matches it on DIBCO 2017
  (74.6 vs 71.5), 2018 (64.7 vs 64.6) and 2009 (70.5 vs 71.9).
- The headline stain result comes from 22 hand-picked images and is shown only
  in a figure (Fig. 13); no table gives those numbers.
- Subjective study: only 18% of the stain samples got the top grade.
- The authors themselves report failure on thin strokes and text breakage
  (Fig. 16).

### The tables have errors

- Bradley scores F = 100 with DRD = 0 on DIBCO 2010.
- The Kittler and Sauvola rows are identical in many columns.
- Every method scores F ≈ 10–12 on DIBCO 2012–2014. That points to a broken
  evaluation, not hard data.
- One row of the observer-grade table (Sample-17) sums to 116%.
- The abstract's "93% precision" appears in no table.

### The method doesn't hold together if read literally

- **Eq. 6 makes level 2 output binary** (255 or 0). Level 3 then splits the
  "distinct grey levels" of that output into four groups, but there are only
  two levels. As written, level 3 does nothing.
- **Eq. 7 makes level 3 output binary too**, so the later re-threshold at
  `mean(g3) / 2` inside the mask also does nothing.
- **The pseudo-colouring pseudocode sets every pixel outside the mask to 0**,
  which would blacken the whole page. This is almost certainly a nesting typo.
- **Eq. 1 is garbled.** It could mean the mean intensity of `Gd` or the mean of
  its distinct grey levels.
- It is unclear which Otsu class (`Iref = 1`) is "text": that depends on the
  sign of `G − (R+B)/2` for ink compared with stain.

Any implementation has to fill these gaps with its own interpretation, and must
document that interpretation.

---

## 2. The plan, section by section

### §1 Dependencies

`opencv-python-headless`, `numpy` and `scikit-image` are all already in
`pyproject.toml`. **Nothing needs adding.**

### §2 Step 1: Pre-Enhancement & Gradient Stain Detector

| Issue | Detail | Suggestion |
|---|---|---|
| PDF pages have no colour | `pipeline.py:159` rasterizes PDF pages with `grayscale=True`, so this step has no RGB channels on PDFs. | Rasterize as RGB when this step is in the chain, or fall back: flag the page and skip. |
| Paper is tinted overall | The samples `1-1/1-2/1-4.jpg` have a mean channel spread of about 78, i.e. uniformly yellowed paper. Otsu on `G − (R+B)/2` may just separate paper from ink rather than stain from paper. | Check visually on the samples before building anything on top of it. |
| 500 px is too small at 300 DPI | On pages of about 2000×2900 px, a single Ottoman letter body plus its dots can exceed 500 px, so the mask would catch text, not stains. | Make the minimum area a fraction of the page area (or scale it with DPI). |
| `record.stain_mask` doesn't exist | `PageRecord` already has `regions: list[Region]` with `kind="stain"` in mind; they are auditable and serialized by `to_dict`. | Emit `Region(kind="stain")` entries plus a flag. Add a small non-serialized field for the raw mask so later steps can read it. |
| Mask source differs from the paper | The paper takes the stain mask from Otsu on `Gd` (§3.6), not on `Id`. | Either is defensible; state which one is used. |
| Destructive by design | Most of the logic overwrites pixels. | Keep this step **non-destructive**: it only detects and annotates, matching "nothing is silently dropped". |

### §2 Step 2: Tri-Level Binarizer

- **Level 1:** "mean pixel intensity of the gradient image" is one reading of
  the garbled Eq. 1. `Gd` contains blacked-out pixels, which drag the mean
  down, so the choice matters. Document it.
- **Level 2:** setting only pixels *above* the median to 255 differs from the
  paper (Eq. 6 also sets the rest to 0). It is the reading that leaves level 3
  something to do, so it is a sensible fix, but label it as an interpretation.
- **Level 3:** with the level-2 fix it has an input to work on. The output is
  binary.
- **It must have `name = "binarize"`.** `run_image_steps` keys on that name to
  snapshot the lossless grayscale into `record.gray`. Without it, `record.gray`
  ends up holding the wrong image.
- **Don't stack it after CLAHE without measuring.** CLAHE equalizes locally and
  flattens the page-wide histogram that these mean, median and partition
  thresholds depend on.

### §2 Step 3: Pseudo-Coloring Ink Rescuer

- Re-thresholding the **original grayscale** inside the mask (rather than `g3`)
  is again a reasonable fix for the paper's no-op.
- Note what it becomes, though: a second local threshold inside stain areas.
  Sauvola already thresholds locally everywhere, so this step only earns its
  place if it measurably beats Sauvola inside stain regions.
- `mean(level_3_output) * 0.5`: level 3 output is mostly 255, so this lands
  around 115–125 regardless of the stain. Consider computing it from the
  grayscale *inside the mask*.
- Pixels outside the mask: the plan leaves them as they are. That is correct.
  Don't copy the paper's pseudocode, which sets them to 0.

### §2 Step 4: Morphological Restorer — the riskiest step for this script

- Ottoman letters are told apart by their **dots** (ب ت ث ن ی). Dilation merges
  dots into letter bodies.
- Dilation also closes the loops of ه و م.
- Bridging to "reconnect broken ligatures" adds joins that shouldn't exist:
  Arabic script breaks deliberately after letters that don't join forward
  (ا د ذ ر ز و).
- The paper only tested this on palm leaf and DIBCO (mostly Latin), not on
  Arabic-script print.
- **Recommendation:** opt-in, **off by default**, with a docstring warning.
  `ComponentFilter` already handles speck cleanup in a script-safe way.

### §3 Example pipeline

The example silently drops CLAHE, denoise, `ConfidenceMap` and
`ComponentFilter`, so faint regions are no longer flagged. Offer it as an
**alternative chain** (e.g. `tri_level_image_steps()` next to
`default_image_steps()`), not as a replacement.

### Housekeeping

- Remove the `[cite: N]` markers; they look like leftovers from whatever tool
  generated the plan.
- `docs/` is untracked and holds the 8.6 MB paper PDF. The paper is CC-BY, so
  committing it is allowed, but consider git-ignoring it and citing the DOI
  instead.

---

## 3. Revised roadmap

0. **Evaluation harness (prerequisite).** The repo has no metrics today. Add
   `ottoparse/eval/` with F-measure, pseudo-F-measure, PSNR and DRD. Run them
   on the printed DIBCO images plus a few hand-labelled Ottoman crops (stained
   and clean). Without this, no claim of improvement can be checked.
1. **`StainDetector`** (new, non-destructive): colour-difference map, Otsu,
   then an area filter scaled to the page. Emits `Region(kind="stain")`, a flag,
   and a mask; falls back gracefully on grayscale input. This is the part of
   the paper most likely to pay off, and it is useful even if the tri-level
   binarizer loses.
2. **`TriLevelBinarizer`** (`name = "binarize"`): an alternative to Sauvola,
   not a replacement. Its docstring states exactly how it reads Eq. 1, 6 and 7.
3. **`StainRescue`** (opt-in): re-threshold the preserved grayscale inside
   stain regions only.
4. **`MorphologicalRestorer`** (opt-in, off by default): small kernel and a
   docstring warning about dots and letter loops.
5. **Benchmark:** the current default chain vs the tri-level chain vs the
   default chain plus `StainDetector`/`StainRescue`. Change the default only if
   the numbers support it, and record the results in `CHANGELOG.md`.

Tests for each step should follow the existing `tests/test_imaging.py` style.
Include synthetic cases: a page with a coloured blot over text, a grayscale-only
input (fallback path), and a dotted glyph surviving `MorphologicalRestorer`
when it is disabled.
