# AFID data audit

Audit of the three archives downloaded from the dataset page
(`Defect_images.7z`, `NODefect_images.7z`, `Mask_images.7z`).

## Counts

| Fabric | Defect-free | Defective |
|---|---|---|
| 00 | 20 | 7 |
| 01 | 20 (21 in the archive) | 12 |
| 02 | 20 | 38 |
| 03 | 20 | 31 |
| 04 | 20 | 10 |
| 05 | 20 | 1 |
| 06 | 20 | 6 |
| 08 | n/a | 0 (1 in the archive) |
| Total | 140 | 105 |

The archives hold 141 defect-free and 106 defective files. The dataset page states
140 and 105. The two extra files are excluded:

- `0018_00_01.png`: a 21st defect-free image for fabric 01 whose name breaks the
  `nnnn_000_ff` pattern.
- `0100_025_08.png`: fabric code 08 does not exist elsewhere in the data, and the image
  has no mask.

## Defect codes (from the dataset page)

| Code | Defect | Images (after exclusion) |
|---|---|---|
| 002 | Broken end | 9 |
| 006 | Broken yarn | 8 |
| 010 | Broken pick | 10 |
| 016 | Weft curling | 3 |
| 019 | Fuzzyball | 39 |
| 022 | Cut selvage | 9 |
| 023 | Crease | 5 |
| 025 | Warp ball | 5 |
| 027 | Knots | 1 |
| 029 | Contamination | 1 |
| 030 | Nep | 14 |
| 036 | Weft crack | 1 |

## Oddities handled by the code

- All images are 4096x256 except `0094_027_05.png` (3796x256); its mask matches.
- 31 files are stored as grayscale with an alpha channel and are loaded as plain grayscale.
- Masks are not binary: most contain anti-aliased grey levels. Thresholding at > 127
  discards about 0.9% of the non-zero mask pixels.
- `0044_019_04` and `0097_030_03` have two masks each (`_mask1`, `_mask2`); the code
  takes their union.
- `0106_010_03_mask.png` is entirely empty. The image is kept in the manifest with
  `mask_empty=True`.
- No exact (byte-level) duplicates were found. Near-duplicates were not checked.

## What this means for evaluation

- Fabric 05 has one defective image, so leave-one-fabric-out cannot assess defect detection
  on it. It can only be used for a false-alarm check on clean images.
- Fabrics 00, 04 and 06 have 6 to 10 defective images; intervals will be wide.
- Fabrics 02 and 03 hold 69 of the 105 defective images, so per-fabric averages are
  preferred over pooled metrics.
- Only fuzzyball has a usable count for per-type results. Other types are reported as
  descriptive counts only.
- Median defect area is about 0.03% of an image, so the patch-labelling threshold is a
  key design choice (Phase 2).

## Capture artefacts found after looking at the images

These were found by inspecting failure cases, not from the file listing, and they change how
results should be read. Reproduce with `scripts/find_margins.py` and
`scripts/artefact_check.py`.

- **Blank margins.** 204 of the 245 strips contain saturated white columns where the camera
  sees past the fabric (up to 30% of a strip, median leading margin 238 px). 15% of patches
  touch one (with a 64 px safety band), including 30 of the 365 defective patches. The
  amount of margin differs between clean and defective strips of the same fabric, so
  margin alone separates them with AUROC 1.00 within fabric 01 and 0.74 within fabric 02.
- **Brightness differs for capture reasons.** In fabrics 00, 01, 02 and 05 the clean strips
  have almost identical mean brightness (standard deviation 0.6 to 2.8 grey levels) while
  defective strips are brighter on average and scatter widely (fabric 01: 119 against 110,
  standard deviation 17). A defect covering about 0.03% of a strip cannot cause that. A
  brightness-only baseline reaches within-fabric AUROC 0.83 on fabric 01 and 0.78 on
  fabric 02 after margin masking. Fabrics 03, 04 and 06 show no such difference.
- **Dark edge bands.** Some strips have a dark band beside the margin or at one end (for
  example `0056_019_06`). These are not detected or masked.

Defects never overlap a blank column, so masking blank columns removes no labelled defect
pixels; only the 64 px safety band reaches defective patches.
