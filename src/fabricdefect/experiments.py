"""The evaluation protocol: three splits x every method, one test evaluation each.

For each fold: fit on train, choose the decision threshold on validation patches only,
score the test patches. Nothing from the test side reaches fitting, hyper-parameter
selection or threshold selection.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .models import NotApplicable, View, method_factories
from .patchdata import PatchData
from .splits import SEED, grouped_kfold, leaky_patch_kfold, leave_one_fabric_out
from .thresholds import best_f1_threshold

SPLITS = ("grouped", "lofo", "leaky")
K_FOLDS = 5


@dataclass
class Fold:
    label: str
    train: np.ndarray  # patch indices
    val: np.ndarray
    test: np.ndarray


@dataclass
class Run:
    scores: np.ndarray  # out-of-fold score for every patch
    thr: np.ndarray  # decision threshold of the fold each patch was tested in
    info: dict = field(default_factory=dict)


def build_folds(pdat: PatchData, k: int = K_FOLDS, seed: int = SEED) -> dict[str, list[Fold]]:
    items = [
        SimpleNamespace(image_id=i, fabric=f, kind=kd)
        for i, f, kd in zip(pdat.image_ids, pdat.fabrics, pdat.kinds, strict=True)
    ]

    def to_patches(ids) -> np.ndarray:
        return pdat.patches_for_images(pdat.image_index(ids))

    folds = {
        "grouped": [
            Fold(s.fold, to_patches(s.train), to_patches(s.val), to_patches(s.test))
            for s in grouped_kfold(items, k=k, seed=seed)
        ],
        "lofo": [
            Fold(s.fold, to_patches(s.train), to_patches(s.val), to_patches(s.test))
            for s in leave_one_fabric_out(items, seed=seed)
        ],
        "leaky": [
            Fold(str(s.fold), s.train, s.val, s.test)
            for s in leaky_patch_kfold(pdat.image_ids[pdat.patch_image].tolist(), k=k, seed=seed)
        ],
    }
    return folds


def make_view(pdat: PatchData, idx: np.ndarray) -> View:
    img = pdat.patch_image[idx]
    return View(
        X=pdat.features[idx],
        y=pdat.patch_label[idx],
        raw_mean=pdat.raw_mean[idx],
        raw_std=pdat.raw_std[idx],
        fabric=pdat.fabrics[img],
        clean_image=pdat.kinds[img] == "clean",
    )


def run_method(
    pdat: PatchData,
    folds: list[Fold],
    factory: Callable,
    threshold_fn: Callable = best_f1_threshold,
    valid: np.ndarray | None = None,
) -> Run:
    """Run one method over all folds of one split. Raises NotApplicable if it cannot run.

    `valid` (one boolean per patch) marks patches the method may use. Invalid patches are
    left out of fitting and threshold selection and, at test time, get a score below the
    decision threshold so they can never raise an alarm.
    """
    n = len(pdat.patch_x)
    scores = np.full(n, np.nan)
    thr = np.full(n, np.nan)
    info: dict = {"fit_seconds": [], "score_ms_per_patch": [], "C": []}
    for fold in folds:
        scorer = factory()
        keep = (lambda i: i) if valid is None else (lambda i: i[valid[i]])
        train, val = make_view(pdat, keep(fold.train)), make_view(pdat, keep(fold.val))
        test_idx = keep(fold.test)
        test = make_view(pdat, test_idx)
        t0 = time.perf_counter()
        scorer.fit(train, val)
        info["fit_seconds"].append(time.perf_counter() - t0)
        if scorer.fixed_threshold is not None:
            t = scorer.fixed_threshold
        else:
            lab = val.y >= 0
            t = threshold_fn(val.y[lab] == 1, scorer.score(val)[lab])  # validation only
        t0 = time.perf_counter()
        scores[test_idx] = scorer.score(test)
        info["score_ms_per_patch"].append((time.perf_counter() - t0) / max(len(test_idx), 1) * 1000)
        thr[fold.test] = t
        if valid is not None:  # unusable patches sit below the threshold and every real score
            scores[fold.test[~valid[fold.test]]] = min(t, scores[test_idx].min()) - 1.0
        if "C" in scorer.info:
            info["C"].append(scorer.info["C"])
    if np.isnan(scores).any() or np.isnan(thr).any():
        raise AssertionError("some patches were never tested")
    return Run(scores, thr, info)


def run_protocol(
    pdat: PatchData,
    progress: Callable[[str], None] | None = None,
    valid: np.ndarray | None = None,
):
    """Every method on every split. Returns {(split, method): Run | None (not applicable)}."""
    folds = build_folds(pdat)
    runs: dict[tuple[str, str], Run | None] = {}
    for split in SPLITS:
        for name, factory in method_factories():
            try:
                runs[(split, name)] = run_method(pdat, folds[split], factory, valid=valid)
            except NotApplicable:
                runs[(split, name)] = None
            if progress:
                progress(
                    f"{split:8s} {name}"
                    + ("  (not applicable)" if runs[(split, name)] is None else "")
                )
    return runs
