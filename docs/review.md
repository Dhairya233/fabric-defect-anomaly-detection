# Review of the evaluation

The project was reviewed from four angles, hostile on purpose. Each criticism below was
tested with an experiment or a test rather than argued, and the outcome is stated even
when it went against the first guess. Experiments are in `scripts/review_checks.py`;
their outputs are in `results/review/`.

## A sceptical reviewer

| Criticism | What was done | Outcome | Action |
|---|---|---|---|
| Margin masking may leave a shortcut: strips with less white margin have more scored patches, so "max over patches" has more chances to fire. | Within each fabric, AUROC of "number of scored patches" alone, then strip scores recomputed from an equal number of random scored patches per strip (`count_matched_image_scores`). | The count confound is real (AUROC 0.96 within fabric 01, 0.64 within fabric 00), but equalising it changes the macro-fabric AUROC by at most 0.014 (mean 0.004) for every method and split, and fabric 01 stays between 0.83 and 1.00 in the grouped and leaky splits (largest change from matching: 0.03). It is not what drives fabric 01. The brightness difference documented in the audit is the better explanation. | Kept as a tested sensitivity function; result recorded in `results/review/count_matched.csv`. |
| Strips of one fabric are not independent, so strip-level intervals are too narrow for claims about new fabrics. | Resampled fabrics instead of strips (6 fabrics, 10,000 draws). | The random forest's unseen-fabric interval widens from 0.44 to 0.62 to 0.34 to 0.72, the SVM's to 0.46 to 0.78. The grouped-to-unseen drop still holds at fabric level for the random forest (+0.30, 0.10 to 0.52; higher in 5 of 6 fabrics) and the SVM (+0.15, 0.05 to 0.23), not for the isolation forest. | "Nothing transfers" reworded to "no evidence of transfer; a modest amount cannot be ruled out". |
| One random fold assignment; the intervals ignore it. | Repeated the grouped split with five fold assignments for three methods. | Macro-fabric AUROC: random forest 0.818 ± 0.017, SVM 0.792 ± 0.048, per-fabric Mahalanobis 0.787 ± 0.004. The method differences are about as large as this noise. | Disclosed in results and on the site; no ranking between those three is claimed. |
| Analysis choices were made after seeing test results. | Listed them. | Margin-masked run, 64 px safety band (set once), the three-fabric subset rule, preferring per-fabric AUROC, the method shown in the local failure gallery. | Disclosed; both runs are reported side by side instead of replacing one with the other. |
| Near-duplicate strips could leak across splits. | Normalised cross-correlation of every within-fabric pair (4,745) against a null from different-fabric pairs. | No pair reaches duplicate level: the maximum is at most 0.86 in every fabric except 06, where the typical pair already correlates at 0.76 (maximum 0.92). Partial overlaps in fabric 06 cannot be excluded by this test. | Recorded; limitation stated. |
| Thresholds may have been chosen on test data. | Existing test spies on the threshold function and checks it receives exactly each fold's validation patches. | Passes. | None needed. |
| Headline numbers may not match the tables. | New test compares the README's and site's key numbers with the CSVs. | It found a real error: the random forest's grouped patch precision is 0.235 (0.2349), which I had written as 0.24. | Corrected to 0.23 everywhere; the test now guards it. |
| Wording stronger than the evidence. | Scanned README, results and site for absolute claims. | "Nothing transfers", "at chance", "show no inflation" and an unlabelled "best" were stronger than the data supports. | All four reworded. |
| Methods are not on equal footing: supervised models use defect labels, per-fabric Mahalanobis is told the fabric, only the SVM has a tuned setting. | Reviewed the protocol. | True, and visible in the method sheet. Not changed. | Stated in the method descriptions. |
| Strip F1 uses a threshold chosen for patch-level F1. | Reviewed the protocol. | True; strip F1 is therefore not the best achievable for any method. | Stated in the results limitations. |
| Results may not reproduce. | Reran the random forest on the grouped split and compared with the cached predictions. | Identical (maximum difference 0.0). | None needed. |

## A senior engineer

- A second copy of the defect-name table lived in `reporting.py`; it could drift from
  `dataset.py`. Now a single source.
- An unused `FABRICS` constant and an unused `family` attribute on every model class were
  removed.
- `assemble_score_map` (the 2-D score map) is only used by tests; the experiments use the
  1-D column version, which carries the same information for full-height patches. It is
  kept as the documented public helper and tested against the 1-D version.
- Every script now fails with a one-line instruction ("Run X first") when a prerequisite is
  missing, instead of a traceback. `prepare_data.py` no longer "succeeds" with zero images
  and warns if the counts differ from the dataset's documented 140 and 105.
- Dependencies were missing from `requirements.txt` and `pyproject.toml` (found when the
  site work started); fixed and verified in a clean environment.

## A stranger on a clean machine

The repository was copied into an empty directory with a fresh virtual environment and
every script was run out of order. Findings and fixes are the script messages above and the
README now stating: create `data/archives/` yourself, run from the repository root, and what
the full run needs (about 400 MB of disk; it ran on one CPU core within 4 GB of memory;
about 15 minutes for features and about 30 minutes per experiment run).

## A recruiter with 60 seconds

The README began with a methods paragraph and a status line. It now opens with one
sentence on what the project is, the stack, and three findings with numbers, then the main
figure. The site gained a visible link to the code and a line naming the stack, and its
headline was softened from "at chance" to "near chance".

## Still open

- Why the isolation forest is below chance on fabric 04 even within the grouped split.
- Dark edge bands next to some margins are not masked.
- Deep pretrained features are not evaluated.
- Six fabrics is a small sample for any claim about "new fabrics".
