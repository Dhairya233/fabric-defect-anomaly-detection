import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fabricdefect.evaluation import (
    bootstrap,
    hist_auroc_ap,
    interval,
    metrics,
    prepare,
)
from fabricdefect.patchdata import PatchData
from fabricdefect.patches import assemble_column_scores, assemble_score_map


def test_hist_metrics_match_sklearn_on_binned_scores():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.3
    s = np.round(rng.normal(size=2000) + y, 1)  # many ties
    bins = np.unique(s)
    idx = np.searchsorted(bins, s)
    pos = np.bincount(idx[y], minlength=len(bins)).astype(float)
    neg = np.bincount(idx[~y], minlength=len(bins)).astype(float)
    auroc, ap = hist_auroc_ap(pos, neg)
    assert np.isclose(auroc, roc_auc_score(y, s))
    assert np.isclose(ap, average_precision_score(y, s), atol=1e-9)


def test_hist_metrics_need_both_classes():
    assert np.isnan(hist_auroc_ap(np.zeros(5), np.ones(5))[0])


def test_column_scores_equal_a_row_of_the_score_map():
    starts = [0, 128, 256, 300]
    scores = np.array([1.0, 2.0, 4.0, 8.0])
    assert np.allclose(
        assemble_column_scores(scores, starts, 556), assemble_score_map(scores, starts, 556)[10]
    )


def _oracle(pdat: PatchData):
    jitter = 0.1 * (np.arange(len(pdat.patch_x)) % 7) / 7  # below the 0.5 threshold
    scores = pdat.patch_label.astype(float).clip(min=0) + jitter
    return scores, np.full(len(scores), 0.5)


def test_oracle_scores_give_perfect_detection(pdat):
    scores, thr = _oracle(pdat)
    m = metrics(prepare(pdat, scores, thr), np.ones(pdat.n_images))
    for k in ("image_auroc", "patch_auroc", "image_f1", "patch_f1"):
        assert np.isclose(m[k], 1.0), k
    # A perfect patch-level oracle still cannot localise below patch resolution: the
    # defect here covers 30 columns inside a much wider flagged region.
    assert 0.6 < m["pixel_auroc"] < 1.0
    assert 0.0 < m["pixel_iou"] < 0.5


def test_constant_scores_are_chance_level(pdat):
    scores, thr = np.zeros(len(pdat.patch_x)), np.ones(len(pdat.patch_x))
    m = metrics(prepare(pdat, scores, thr), np.ones(pdat.n_images))
    assert np.isclose(m["image_auroc"], 0.5) and m["image_recall"] == 0.0


def test_bootstrap_is_paired_across_keys_and_resamples_images(pdat):
    scores, thr = _oracle(pdat)
    prep = prepare(pdat, scores, thr)
    point, boot = bootstrap({("a", "m"): prep, ("b", "m"): prep}, n_boot=30, seed=1)
    assert np.array_equal(
        boot[("a", "m")]["patch_ap"], boot[("b", "m")]["patch_ap"], equal_nan=True
    )
    assert point[("a", "m")]["image_auroc"] == 1.0
    lo, hi, valid = interval(boot[("a", "m")]["image_auroc"])
    assert lo <= hi and valid > 0


def test_interval_refuses_too_few_samples():
    assert np.isnan(interval(np.array([0.1, np.nan]))[0])


def test_patchdata_roundtrip(tmp_path, pdat):
    pdat.save(tmp_path / "p.npz")
    back = PatchData.load(tmp_path / "p.npz")
    assert np.array_equal(back.features, pdat.features)
    assert back.n_images == pdat.n_images
    assert np.array_equal(back.patch_image, pdat.patch_image)
