# Patching, labels and splits

## Patches

Strips are 4096x256, so patches are 256x256 squares that slide along the width with a
stride of 128 px (50% overlap): 31 patches per full strip, 7,593 patches over the 245
images. A final patch is aligned to the right edge when the width is not a multiple of the
stride (the 3796 px strip gets 28 patches), so every pixel is covered.

Overlap matters because defects are small. Without it, a defect straddling a patch
boundary could be cut in two and look weak in both halves. With 50% overlap, any defect up
to 128 px wide lies entirely inside at least one patch. Patch scores are mapped back to
the strip by averaging wherever patches overlap (`assemble_score_map`).

## Patch labels

A patch is DEFECTIVE if it contains at least `MIN_DEFECT_PX = 5` mask pixels, CLEAN if it
contains none, and AMBIGUOUS otherwise. Ambiguous patches are left out of patch-level
training and scoring; pixel-level scoring always uses the full masks.

The threshold is small on purpose. The median defect covers about 280 pixels, so a rule
like "at least 10% of the patch" would erase nearly all positives.
`results/patch_label_threshold.csv` (regenerate with
`python scripts/label_threshold_study.py`) shows the effect on the 105 defective images:

| min defect px | defective images with no positive patch | positive patches | ambiguous patches |
|---|---|---|---|
| 1 | 1 | 366 | 0 |
| 5 | 1 | 365 | 1 |
| 10 | 4 | 357 | 9 |
| 25 | 28 | 304 | 62 |
| 100 | 53 | 248 | 118 |
| 2000 | 98 | 60 | 306 |

At 25 px more than a quarter of the defective images would have no defective patch at all.
The single image missed at 1 and 5 px is `0106_010_03`, whose mask is empty. Five pixels
keeps all the real defects and ignores one-pixel mask specks.

## Splits

All splits are by whole image, written so that a test image's patches never appear on the
train or validation side.

- **Grouped 5-fold** (`grouped_kfold`): stratified by (fabric, clean/defective), so each
  fold's test side has 49 images, 21 of them defective.
- **Leave-one-fabric-out** (`leave_one_fabric_out`): the held-out fabric is the entire test
  side. Validation images come from the other fabrics.
- **Leaky patch k-fold** (`leaky_patch_kfold`): random patches, ignoring image identity.
  It exists only to measure inflation. In the real data every fold has patches of
  essentially every test image on the training side too (245 of 245 test images in four
  folds, 244 in one), against zero for the two proper splits.

Each split has three roles. Train fits models; anomaly models use only the defect-free
images in it. Validation chooses thresholds (`best_f1_threshold` takes validation arrays
only). Test is evaluated once per configuration.

Fabric 05 has a single defective image. It appears as a test image in one grouped fold and
in the 05 leave-one-fabric-out split, where only false alarms can be measured.

## What the tests assert

- Every image is tested exactly once across grouped folds and never shares a side.
- A leave-one-fabric-out split's test side is exactly that fabric, and the fabric is absent
  from train and validation.
- Validation images are never test images.
- The leaky split shares images across sides (it is supposed to) and the grouped split
  shares none.
- The threshold function accepts only validation arguments. A test that the experiment
  runner never passes test data to it is added with the runner.
