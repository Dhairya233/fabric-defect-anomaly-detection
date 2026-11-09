"""Turn out-of-fold predictions into the result tables written to results/."""

from __future__ import annotations

import platform

import numpy as np
import pandas as pd

from .dataset import DEFECT_NAMES as _DEFECT_NAMES
from .evaluation import (
    METRICS,
    bootstrap,
    interval,
    prepare,
    subset_image_metrics,
)
from .experiments import Run
from .patchdata import PatchData

HEADLINE = [
    "image_auroc",
    "image_ap",
    "image_f1",
    "image_macro_fabric_auroc",
    "patch_auroc",
    "patch_ap",
    "patch_f1",
    "pixel_auroc",
    "pixel_ap",
]
COMPARISONS = [("leaky", "grouped"), ("leaky", "lofo"), ("grouped", "lofo")]
DEFECT_NAMES = {code: name for code, name in _DEFECT_NAMES.items() if code != 0}


def build_tables(
    pdat: PatchData,
    runs: dict[tuple[str, str], Run | None],
    n_boot: int = 1000,
    seed: int = 0,
    n_boot_subset: int = 500,
    valid: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    preps = {k: prepare(pdat, r.scores, r.thr, valid) for k, r in runs.items() if r is not None}
    point, boot = bootstrap(preps, n_boot, seed)

    n_img = pdat.n_images
    n_def = int((pdat.kinds == "defective").sum())
    ok = np.ones(len(pdat.patch_x), bool) if valid is None else valid
    n_lab = int(((pdat.patch_label >= 0) & ok).sum())
    n_pos = int(((pdat.patch_label == 1) & ok).sum())

    rows = []
    for (split, method), _ in runs.items():
        if (split, method) not in preps:
            rows.append({"split": split, "method": method, "metric": "not applicable"})
            continue
        for m in METRICS:
            lo, hi, valid = interval(boot[(split, method)][m])
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "metric": m,
                    "value": point[(split, method)][m],
                    "ci_low": lo,
                    "ci_high": hi,
                    "valid_resamples": valid,
                    "n_images": n_img,
                    "n_defective_images": n_def,
                    "n_labelled_patches": n_lab,
                    "n_defective_patches": n_pos,
                }
            )
    overall = pd.DataFrame(rows)

    gap_rows = []
    methods = sorted({m for (_, m) in preps})
    for method in methods:
        for a, b in COMPARISONS:
            if (a, method) not in preps or (b, method) not in preps:
                continue
            for m in HEADLINE:
                diff = boot[(a, method)][m] - boot[(b, method)][m]
                lo, hi, valid = interval(diff)
                gap_rows.append(
                    {
                        "method": method,
                        "comparison": f"{a} minus {b}",
                        "metric": m,
                        "difference": point[(a, method)][m] - point[(b, method)][m],
                        "ci_low": lo,
                        "ci_high": hi,
                        "valid_resamples": valid,
                    }
                )
    gap = pd.DataFrame(gap_rows)

    defective = pdat.kinds == "defective"
    fab_rows, type_rows = [], []
    for (split, method), prep in preps.items():
        for f in np.unique(pdat.fabrics):
            on_f = pdat.fabrics == f
            r = subset_image_metrics(prep, defective & on_f, ~defective & on_f, n_boot_subset, seed)
            fab_rows.append({"split": split, "method": method, "fabric": f, **r})
        for code in sorted(DEFECT_NAMES):
            pos = pdat.codes == code
            r = subset_image_metrics(prep, pos, ~defective, n_boot_subset, seed)
            type_rows.append(
                {
                    "split": split,
                    "method": method,
                    "defect_code": code,
                    "defect": DEFECT_NAMES[code],
                    **r,
                }
            )
    by_fabric = pd.DataFrame(fab_rows)
    by_type = pd.DataFrame(type_rows)
    by_fabric["reliable"] = by_fabric["n_defective"] >= 5
    by_type["reliable"] = by_type["n_defective"] >= 5

    cost_rows = []
    extract_s = float(pdat.extract_seconds.mean())
    patches_per_strip = float(np.diff(pdat.patch_offsets).mean())
    for (split, method), run in runs.items():
        if split != "grouped" or run is None:
            continue
        score_ms = float(np.mean(run.info["score_ms_per_patch"]))
        needs_features = not method.startswith("M0")
        per_strip_ms = score_ms * patches_per_strip + (extract_s * 1000 if needs_features else 0)
        cost_rows.append(
            {
                "method": method,
                "fit_seconds_per_fold": float(np.mean(run.info["fit_seconds"])),
                "score_ms_per_patch": score_ms,
                "feature_extraction_s_per_strip": extract_s if needs_features else 0.0,
                "total_ms_per_strip": per_strip_ms,
                "cpu": platform.processor() or platform.machine(),
                "cpu_cores_available": 1,
            }
        )
    cost = pd.DataFrame(cost_rows)
    return {
        "metrics_overall": overall,
        "leakage_gap": gap,
        "by_fabric": by_fabric,
        "by_defect_type": by_type,
        "cost": cost,
    }
