# Results

All numbers come from `results/` and `results/margin_masked/`. Intervals are 95%
bootstrap intervals from resampling whole images (1,000 resamples); every image is tested
exactly once per split. Seed 0 throughout.

Two runs are reported:

- **As-is** (`results/`): every patch is used.
- **Margin-masked** (`results/margin_masked/`): patches that touch a blank margin are left
  out of fitting and evaluation and cannot raise an alarm (15% of patches; see
  [data_audit.md](data_audit.md)). This is the run to read first, because the margins turned
  out to distort the as-is numbers.

Charts for both runs are in `results/margin_masked/figures/` (margins left out) and
`results/figures/` (as-is): the leakage comparison, the per-fabric AUROC heatmap and recall by
defect type.

## What to trust

Strip-level AUROC pooled over all 245 strips mostly rewards recognising the fabric:
fabrics 02 and 03 hold 69 of the 105 defective strips, and they have the highest raw
contrast. That is why the raw-contrast baseline scores 0.756 pooled (margin-masked) but
only 0.590 when AUROC is computed within each fabric and averaged ("macro-fabric AUROC").
Prefer the macro-fabric numbers.

Within fabrics 00, 01, 02 and 05 the clean and defective strips also differ in overall
brightness, which a defect covering 0.03% of a strip cannot cause (see the audit). High
scores on those fabrics are therefore not evidence of defect detection: a brightness
baseline alone reaches within-fabric AUROC 0.83 on fabric 01. Fabrics 03, 04 and 06 show no
such difference, so they are the cleanest test. The rule that selects them was written after
looking at the data, so this subset is a sensitivity analysis, not a pre-registered test.

## Strip-level detection, margin-masked

Mean of per-fabric image AUROC (0.5 = chance).

**Fabrics 03, 04, 06** (47 defective strips, no brightness difference):

| Method | grouped by image | unseen fabric | leaky patch split |
|---|---|---|---|
| Random forest | 0.74 [0.61, 0.85] | 0.59 [0.44, 0.72] | 0.84 [0.76, 0.91] |
| SVM (RBF) | 0.71 [0.58, 0.83] | 0.50 [0.36, 0.62] | 0.90 [0.82, 0.95] |
| Mahalanobis, per fabric | 0.62 [0.51, 0.74] | not applicable | 0.64 [0.52, 0.77] |
| Mahalanobis, global | 0.58 [0.46, 0.70] | 0.51 [0.40, 0.62] | 0.55 [0.44, 0.67] |
| Isolation forest | 0.53 [0.43, 0.64] | 0.34 [0.23, 0.46] | 0.51 [0.40, 0.62] |
| One-class SVM | 0.45 [0.34, 0.56] | 0.46 [0.35, 0.58] | 0.42 [0.32, 0.53] |
| Raw contrast | 0.45 [0.33, 0.57] | 0.45 [0.33, 0.57] | 0.45 [0.33, 0.57] |

**All fabrics with at least 5 defective strips** (00, 01, 02, 03, 04, 06):

| Method | grouped by image | unseen fabric | leaky patch split |
|---|---|---|---|
| Random forest | 0.82 [0.75, 0.90] | 0.52 [0.44, 0.62] | 0.89 [0.83, 0.95] |
| SVM (RBF) | 0.75 [0.68, 0.86] | 0.60 [0.52, 0.71] | 0.91 [0.86, 0.96] |
| Mahalanobis, per fabric | 0.79 [0.72, 0.89] | not applicable | 0.82 [0.75, 0.91] |
| Mahalanobis, global | 0.75 [0.68, 0.85] | 0.57 [0.46, 0.65] | 0.73 [0.66, 0.84] |
| Raw contrast | 0.59 [0.49, 0.69] | 0.59 [0.49, 0.69] | 0.59 [0.49, 0.69] |

## Findings

1. **No evidence of transfer to an unseen fabric.** With a whole fabric held out, no method
   is clearly above chance on the cleanest subset (best of eight: random forest 0.59,
   interval 0.44 to 0.72), and the isolation forest is below it (0.34, interval 0.23 to
   0.46). Scores from the grouped split do not predict behaviour on a new fabric. With only
   six fabrics the evidence is thin in both directions: resampling fabrics instead of
   strips widens the random forest's unseen-fabric interval to 0.34 to 0.72 and the SVM's
   to 0.46 to 0.78, so a modest amount of transfer cannot be ruled out. The drop from the
   grouped split does hold up at the fabric level for the random forest (+0.30, interval
   0.10 to 0.52, higher in 5 of 6 fabrics) and the SVM (+0.15, 0.05 to 0.23), but not for
   the isolation forest (-0.03 to 0.37). See [review.md](review.md).
2. **Leaky patch splits inflate the supervised models, not the anomaly detectors.** Paired
   leaky-minus-grouped differences (margin-masked):

   | Method | patch average precision | macro-fabric image AUROC |
   |---|---|---|
   | Random forest | +0.44 [0.27, 0.53] (0.235 to 0.679) | +0.07 [0.02, 0.12] |
   | SVM | +0.23 [0.11, 0.32] (0.276 to 0.504) | +0.16 [0.07, 0.23] |
   | Mahalanobis, global | -0.01 [-0.06, 0.00] | -0.02 [-0.05, 0.01] |
   | Mahalanobis, per fabric | +0.01 [0.00, 0.02] | +0.03 [-0.01, 0.06] |

   Localisation metrics suffer most, because overlapping patches of one defect land on both
   sides of the split. Anomaly detectors only ever train on defect-free strips, so there is
   no defect to leak, and their differences are small (per-fabric Mahalanobis +0.03 in
   macro-fabric AUROC, interval -0.01 to 0.06).
3. **The as-is numbers were inflated by the margins.** The best as-is model, per-fabric
   Mahalanobis, scores 0.929 [0.896, 0.959] pooled; masked it scores 0.873 [0.82, 0.92], and
   its macro-fabric AUROC is 0.79. The isolation forest was below chance as-is (0.339) and
   is 0.653 [0.58, 0.72] once margin patches are removed, consistent with margin patches
   looking anomalous and clean strips having more margin. It is still below chance on
   fabric 04 (0.28), which I have not explained.
4. **A fixed threshold is often useless.** An image F1 of exactly 0.600 is what flagging
   every strip scores (105 defective of 245). On the unseen-fabric split three methods land at
   0.596 to 0.611 (margin-masked), meaning the threshold chosen on seen fabrics flags
   essentially everything.
5. **Trivial baselines are unreliable in both directions.** Raw contrast is below chance
   as-is (0.464) and above chance masked (0.756 pooled), purely because of margins and
   fabric mix. This is why baselines are reported next to every method.

## Localisation and defect types

- **Pixel metrics are capped by design.** Patches are 256 px wide and the median defect
  covers about 280 pixels, so even a perfect patch-level detector cannot localise below
  patch resolution. Pixel IoU is below 0.05 for every method; the tables report it, but it
  is a statement about resolution, not about the models.
- **Defect types.** Only fuzzyball (39 strips) and nep (14) have enough examples for a
  usable per-type recall (broken pick has 10); nine of the twelve types have fewer than 10
  strips and three have one. `results/by_defect_type.csv` flags types with fewer than 5 strips as not reliable.

## Cost (single CPU core, x86_64 sandbox)

Feature extraction is 3.17 s per strip and dominates everything else. Once features exist,
scoring a strip takes about 0.1 ms (Mahalanobis) to 5 ms (SVM). Fitting takes 6.3 s per
fold for the random forest and under 3 s for every other method. The baselines need no
features and cost effectively nothing. See `results/cost.csv`. These timings come from one
core of a sandbox machine and are not a deployment benchmark.

## Limitations

- One dataset, 245 strips, 7 fabrics, and only 105 defective strips; fabric 05 has one.
- The intervals cover which test images happened to be drawn. They do not cover the random
  assignment of images to folds. Repeating the grouped split with five fold assignments gave
  a macro-fabric AUROC of 0.818 ± 0.017 for the random forest (range 0.80 to 0.84),
  0.792 ± 0.048 for the SVM (0.73 to 0.84) and 0.787 ± 0.004 for per-fabric
  Mahalanobis. The differences between those three methods are about as large as this
  noise, so their ranking is not reliable.
- Strips of one fabric come from the same material, so they are not independent. The
  strip-level intervals condition on these seven fabrics; claims about new fabrics rest on
  six or seven data points.
- Analysis choices made after seeing results: the margin-masked run, the 64 px safety band
  (set once, not tuned), the artefact-free subset rule, preferring the per-fabric AUROC over
  the pooled one, and which method's failures the local gallery shows. All are reported
  alongside the unmasked numbers rather than replacing them.
- Strip-level F1 uses the decision threshold chosen for patch-level F1 on validation
  patches, so it is not the best achievable strip-level F1 for any method.
- Margin masking removes patches with a saturated margin plus a 64 px safety band. The dark
  edge bands next to some margins (for example strip `0056_019_06`) are not all removed, and
  30 of 365 defective patches fall inside the masked zone.
- The brightness difference between clean and defective strips cannot be removed by
  masking; it is only avoided by looking at fabrics where it is absent.
- Patch-based image decisions flag a strip if any patch crosses the threshold, so false
  alarms grow with the number of patches.
- Deep features (a pretrained network, in the PatchCore style) are not evaluated: I could
  not download pretrained weights in the environment I used. They are the most important
  missing comparison.
