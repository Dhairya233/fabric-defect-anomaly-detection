"""Metrics and bootstrap intervals.

Uncertainty comes from resampling whole images (with replacement), never patches: patches
of one strip are not independent, so patch-level resampling would give intervals that are
far too narrow. Resampling is done with per-image integer weights, which is equivalent to
drawing a bootstrap sample but lets every metric reuse the same machinery. The same
weights are used for every split of one method, so differences between splits get a
proper paired interval.

Predictions are pooled out-of-fold: every image is in the test side of exactly one fold.
Pooling mixes score scales across folds, which is why per-fabric tables are reported too.

Localisation is exact and cheap because patches span the strip height: a score map is
constant down each column, so pixel metrics reduce to per-column counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .patchdata import PatchData
from .patches import PATCH_SIZE, assemble_column_scores

N_BINS = 1000
MIN_DEFECTIVE_FOR_FABRIC_AVERAGE = 5


def hist_auroc_ap(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """AUROC and average precision from score histograms (bins in ascending score order)."""
    p, n = pos.sum(), neg.sum()
    if p <= 0 or n <= 0:
        return np.nan, np.nan
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    tpr = np.r_[0.0, tp / p]
    fpr = np.r_[0.0, fp / n]
    auroc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    precision = tp / np.maximum(tp + fp, 1e-12)
    ap = float(np.sum((tpr[1:] - tpr[:-1]) * precision))
    return auroc, ap


def _auroc(y, s, w) -> float:
    if w[y].sum() == 0 or w[~y].sum() == 0:
        return np.nan
    return float(roc_auc_score(y, s, sample_weight=w))


def _ap(y, s, w) -> float:
    if w[y].sum() == 0:
        return np.nan
    return float(average_precision_score(y, s, sample_weight=w))


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp > 0 else np.nan
    rec = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else np.nan
    return prec, rec, f1


@dataclass
class Prepared:
    """Everything needed to score one (split, method), computed once."""

    pdat: PatchData
    img_defective: np.ndarray  # (n_img,) bool
    img_score: np.ndarray
    img_pred: np.ndarray
    patch_img: np.ndarray  # image index of each labelled patch
    patch_y: np.ndarray
    patch_s: np.ndarray
    patch_pred: np.ndarray
    pix_pos: np.ndarray  # (n_img, N_BINS) defect pixels per score bin
    pix_neg: np.ndarray
    pix_tp: np.ndarray  # (n_img,) at the decision threshold
    pix_fp: np.ndarray
    pix_fn: np.ndarray


def prepare(
    pdat: PatchData, scores: np.ndarray, thr: np.ndarray, valid: np.ndarray | None = None
) -> Prepared:
    """`valid` restricts patch-level metrics to usable patches (see margins.py)."""
    n = pdat.n_images
    pred = scores >= thr
    img_score = np.array([scores[pdat.patches_of(i)].max() for i in range(n)])
    img_pred = np.array([pred[pdat.patches_of(i)].any() for i in range(n)])

    cols, margins = [], []
    for i in range(n):
        sl = pdat.patches_of(i)
        starts = pdat.patch_x[sl].tolist()
        cols.append(assemble_column_scores(scores[sl], starts, int(pdat.widths[i])))
        margins.append(assemble_column_scores(scores[sl] - thr[sl], starts, int(pdat.widths[i])))
    pooled = np.concatenate(cols)
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    nb = len(edges) + 1
    pos = np.zeros((n, N_BINS))
    neg = np.zeros((n, N_BINS))
    tp = np.zeros(n)
    fp = np.zeros(n)
    fn = np.zeros(n)
    for i in range(n):
        d = pdat.col_defect[pdat.columns_of(i)].astype(float)
        b = np.searchsorted(edges, cols[i], side="right")
        pos[i, :nb] = np.bincount(b, weights=d, minlength=nb)
        neg[i, :nb] = np.bincount(b, weights=PATCH_SIZE - d, minlength=nb)
        hit = margins[i] >= 0
        tp[i], fp[i], fn[i] = d[hit].sum(), (PATCH_SIZE - d)[hit].sum(), d[~hit].sum()

    lab = pdat.patch_label >= 0
    if valid is not None:
        lab = lab & valid
    return Prepared(
        pdat=pdat,
        img_defective=pdat.kinds == "defective",
        img_score=img_score,
        img_pred=img_pred,
        patch_img=pdat.patch_image[lab],
        patch_y=pdat.patch_label[lab] == 1,
        patch_s=scores[lab],
        patch_pred=pred[lab],
        pix_pos=pos,
        pix_neg=neg,
        pix_tp=tp,
        pix_fp=fp,
        pix_fn=fn,
    )


METRICS = [
    "image_auroc",
    "image_ap",
    "image_precision",
    "image_recall",
    "image_f1",
    "image_macro_fabric_auroc",
    "patch_auroc",
    "patch_ap",
    "patch_precision",
    "patch_recall",
    "patch_f1",
    "pixel_auroc",
    "pixel_ap",
    "pixel_iou",
]


def metrics(prep: Prepared, w: np.ndarray) -> dict[str, float]:
    """All metrics under image weights w (all ones = the point estimate)."""
    y, s, pr = prep.img_defective, prep.img_score, prep.img_pred
    out = {"image_auroc": _auroc(y, s, w), "image_ap": _ap(y, s, w)}
    tp = float((w * (y & pr)).sum())
    fp = float((w * (~y & pr)).sum())
    fn = float((w * (y & ~pr)).sum())
    out["image_precision"], out["image_recall"], out["image_f1"] = _prf(tp, fp, fn)

    fab = prep.pdat.fabrics
    per_fab = []
    for f in np.unique(fab):
        m = fab == f
        if (w * (y & m)).sum() >= MIN_DEFECTIVE_FOR_FABRIC_AVERAGE and (w * (~y & m)).sum() > 0:
            per_fab.append(_auroc(y[m], s[m], w[m]))
    out["image_macro_fabric_auroc"] = float(np.nanmean(per_fab)) if per_fab else np.nan

    pw = w[prep.patch_img]
    py, ps, pp = prep.patch_y, prep.patch_s, prep.patch_pred
    out["patch_auroc"], out["patch_ap"] = _auroc(py, ps, pw), _ap(py, ps, pw)
    tp = float((pw * (py & pp)).sum())
    fp = float((pw * (~py & pp)).sum())
    fn = float((pw * (py & ~pp)).sum())
    out["patch_precision"], out["patch_recall"], out["patch_f1"] = _prf(tp, fp, fn)

    out["pixel_auroc"], out["pixel_ap"] = hist_auroc_ap(w @ prep.pix_pos, w @ prep.pix_neg)
    tp, fp, fn = w @ prep.pix_tp, w @ prep.pix_fp, w @ prep.pix_fn
    out["pixel_iou"] = float(tp / (tp + fp + fn)) if tp + fp + fn > 0 else np.nan
    return out


def bootstrap(preps: dict[tuple[str, str], Prepared], n_boot: int, seed: int) -> tuple[dict, dict]:
    """Point estimates and bootstrap distributions for every (split, method).

    The same image resamples are used everywhere, so differences between splits can be
    given paired intervals. Returns (point, boot) with boot[key][metric] of shape (n_boot,).
    """
    rng = np.random.default_rng(seed)
    n = next(iter(preps.values())).pdat.n_images
    point = {k: metrics(p, np.ones(n)) for k, p in preps.items()}
    boot = {k: {m: np.full(n_boot, np.nan) for m in METRICS} for k in preps}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for k, p in preps.items():
            for m, v in metrics(p, w).items():
                boot[k][m][b] = v
    return point, boot


def interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float, int]:
    """Percentile interval over the valid (non-NaN) resamples, and how many were valid."""
    ok = samples[~np.isnan(samples)]
    if len(ok) < 20:
        return np.nan, np.nan, len(ok)
    a = (1 - level) / 2
    return float(np.quantile(ok, a)), float(np.quantile(ok, 1 - a)), len(ok)


def subset_image_metrics(
    prep: Prepared, pos_mask: np.ndarray, neg_mask: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Image-level AUROC and recall for positives `pos_mask` against negatives `neg_mask`
    (boolean over images), with intervals from resampling those images."""
    idx = np.flatnonzero(pos_mask | neg_mask)
    y = pos_mask[idx]
    s, pr = prep.img_score[idx], prep.img_pred[idx]

    def one(w):
        return (
            _auroc(y, s, w),
            float((w * (y & pr)).sum() / max((w * y).sum(), 1e-12)) if (w * y).sum() else np.nan,
            float((w * (~y & pr)).sum() / max((w * ~y).sum(), 1e-12)) if (w * ~y).sum() else np.nan,
        )

    point = one(np.ones(len(idx)))
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            one(np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(float))
            for _ in range(n_boot)
        ]
    )
    out = {"n_defective": int(y.sum()), "n_clean": int((~y).sum())}
    for j, name in enumerate(("auroc", "recall", "false_alarm_rate")):
        lo, hi, _ = interval(draws[:, j])
        out[name], out[f"{name}_lo"], out[f"{name}_hi"] = point[j], lo, hi
    return out


def count_matched_image_scores(
    pdat: PatchData,
    scores: np.ndarray,
    valid: np.ndarray,
    m: int | None = None,
    n_draws: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Strip score = expected maximum over a random subset of exactly `m` scored patches.

    A strip's maximum patch score depends on how many patches were scored, and after margin
    masking that number differs between clean and defective strips of the same fabric
    (more scored patches means more chances to exceed any level). Giving every strip the
    same number of patches removes that dependence. `m` defaults to the smallest number of
    scored patches over all strips.
    """
    rng = np.random.default_rng(seed)
    counts = np.array([int(valid[pdat.patches_of(i)].sum()) for i in range(pdat.n_images)])
    m = int(counts.min()) if m is None else m
    if (counts < m).any():
        raise ValueError(f"some strips have fewer than {m} scored patches")
    out = np.zeros(pdat.n_images)
    for i in range(pdat.n_images):
        s = scores[pdat.patches_of(i)][valid[pdat.patches_of(i)]]
        pick = np.argsort(rng.random((n_draws, len(s))), axis=1)[:, :m]
        out[i] = s[pick].max(axis=1).mean()
    return out
