"""Turn the committed result tables and the cached predictions into the small JSON files
the results site reads (site/data/). Needs data/cache/ from the experiment runs, so the
JSON is committed and the site itself never needs the dataset.

No dataset pixels are written: strips are drawn as score tracks and ground-truth extents.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fabricdefect.cli import require
from fabricdefect.patchdata import PatchData
from fabricdefect.reporting import DEFECT_NAMES

N_BINS = 32
MASKED_RUN = "results/margin_masked"
AS_IS_RUN = "results"
METHOD_KEYS = {
    "M0 raw contrast": "contrast",
    "M0 brightness deviation": "brightness",
    "M1 random forest": "rf",
    "M1 SVM (RBF)": "svm",
    "M2 Mahalanobis (global)": "maha_global",
    "M2 Mahalanobis (per fabric)": "maha_fabric",
    "M2 isolation forest (global)": "iforest",
    "M2 one-class SVM (global)": "ocsvm",
    "M0 always clean": "always_clean",
}
TABLE_METRICS = [
    "image_macro_fabric_auroc",
    "image_auroc",
    "image_ap",
    "image_f1",
    "patch_ap",
    "pixel_auroc",
    "pixel_iou",
]
GAP_METRICS = ["image_macro_fabric_auroc", "image_auroc", "patch_ap"]


def clean(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def run_tables(folder: str) -> dict:
    o = pd.read_csv(f"{folder}/metrics_overall.csv")
    na = o[o.metric == "not applicable"]
    o = o[o.metric != "not applicable"]
    table = {}
    for (method, split), g in o.groupby(["method", "split"]):
        row = {}
        for m in TABLE_METRICS:
            x = g[g.metric == m]
            if len(x):
                x = x.iloc[0]
                row[m] = [r(x.value), r(x.ci_low), r(x.ci_high)]
        table.setdefault(METHOD_KEYS[method], {})[split] = row
    gap = pd.read_csv(f"{folder}/leakage_gap.csv")
    gap = gap[gap.metric.isin(GAP_METRICS)]
    gaps = {}
    for _, x in gap.iterrows():
        gaps.setdefault(METHOD_KEYS[x.method], {}).setdefault(x.comparison, {})[x.metric] = [
            r(x.difference),
            r(x.ci_low),
            r(x.ci_high),
        ]
    first = o.iloc[0]
    return {
        "table": table,
        "gaps": gaps,
        "not_applicable": sorted(
            {f"{s}:{METHOD_KEYS[m]}" for s, m in zip(na.split, na.method, strict=True)}
        ),
        "counts": {
            "images": int(first.n_images),
            "defective_images": int(first.n_defective_images),
            "labelled_patches": int(first.n_labelled_patches),
            "defective_patches": int(first.n_defective_patches),
        },
    }


def runs_of_defect(col_defect: np.ndarray, merge_gap: int = 32) -> list[list[int]]:
    xs = np.flatnonzero(col_defect > 0)
    if len(xs) == 0:
        return []
    runs, start, prev = [], xs[0], xs[0]
    for x in xs[1:]:
        if x - prev > merge_gap:
            runs.append([int(start), int(prev) + 1])
            start = x
        prev = x
    runs.append([int(start), int(prev) + 1])
    return runs


def strips_json(pdat: PatchData, predictions: Path, margin_flags: Path) -> dict:
    z = np.load(predictions)
    invalid = np.load(margin_flags)
    methods = [
        m
        for m in sorted({k.split("|")[1] for k in z.files if k.startswith("grouped|")})
        if m != "M0 always clean"
    ]
    # per-method display scale: 95th percentile of |margin| over usable patches
    scale = {}
    for m in methods:
        mg = z[f"grouped|{m}|scores"] - z[f"grouped|{m}|thr"]
        scale[m] = float(np.quantile(np.abs(mg[~invalid]), 0.95)) or 1.0

    strips = []
    for i in range(pdat.n_images):
        sl = pdat.patches_of(i)
        w = int(pdat.widths[i])
        starts = pdat.patch_x[sl]
        ok = ~invalid[sl]
        edges = np.linspace(0, w, N_BINS + 1).astype(int)
        entry = {
            "id": str(pdat.image_ids[i]),
            "fabric": str(pdat.fabrics[i]),
            "defective": bool(pdat.kinds[i] == "defective"),
            "code": int(pdat.codes[i]),
            "defect": DEFECT_NAMES.get(int(pdat.codes[i]), "none"),
            "width": w,
            "gt": [[a / w, b / w] for a, b in runs_of_defect(pdat.col_defect[pdat.columns_of(i)])],
            "scores": {},
        }
        entry["gt"] = [[round(a, 4), round(b, 4)] for a, b in entry["gt"]]
        masked = None
        for m in methods:
            margin = z[f"grouped|{m}|scores"][sl] - z[f"grouped|{m}|thr"][sl]
            acc, cnt = np.zeros(w), np.zeros(w)
            for mg, x, good in zip(margin, starts, ok, strict=True):
                if good:
                    acc[x : x + 256] += mg
                    cnt[x : x + 256] += 1
            col = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
            vals, mk = [], []
            for a, b in zip(edges[:-1], edges[1:], strict=True):
                seg = col[a:b]
                if np.isnan(seg).all():
                    vals.append(None)
                    mk.append(1)
                else:
                    vals.append(int(round(100 * np.clip(np.nanmean(seg) / scale[m], -1, 1))))
                    mk.append(0)
            masked = mk
            flagged = bool((margin[ok] >= 0).any()) if ok.any() else False
            entry["scores"][METHOD_KEYS[m]] = {"v": vals, "flag": flagged}
        entry["masked"] = [j for j, v in enumerate(masked) if v]
        strips.append(entry)
    return {
        "bins": N_BINS,
        "methods": [METHOD_KEYS[m] for m in methods],
        "scale_note": "value = margin over threshold / 95th percentile of |margin|, clipped to +-1 "
        "(x100); 0 is the decision threshold",
        "strips": strips,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("site/data"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for need, hint in (
        ("data/patch_data.npz", "Run python scripts/extract_features.py first."),
        (
            "data/cache/predictions_margin.npz",
            "Run the margin-masked experiment first (see README).",
        ),
        ("data/cache/margin_patch.npy", "Run python scripts/find_margins.py first."),
        ("results/margin_masked/artefact_check.csv", "Run python scripts/artefact_check.py first."),
    ):
        require(need, hint)
    pdat = PatchData.load()
    strips = strips_json(
        pdat, Path("data/cache/predictions_margin.npz"), Path("data/cache/margin_patch.npy")
    )
    (args.out / "strips.json").write_text(
        json.dumps(strips, separators=(",", ":"), allow_nan=False)
    )

    art = pd.read_csv(f"{MASKED_RUN}/artefact_check.csv")
    sub = pd.read_csv(f"{MASKED_RUN}/artefact_free_subset.csv")
    results = {
        "runs": {"masked": run_tables(MASKED_RUN), "as_is": run_tables(AS_IS_RUN)},
        "artefact_check": json.loads(art.replace({np.nan: None}).to_json(orient="records")),
        "artefact_free_subset": {
            f"{s}:{METHOD_KEYS[m]}": [r(v), r(lo), r(hi)]
            for s, m, v, lo, hi in zip(
                sub.split,
                sub.method,
                sub.mean_per_fabric_image_auroc,
                sub.ci_low,
                sub.ci_high,
                strict=True,
            )
        },
        "run_info": json.loads(Path(f"{MASKED_RUN}/run_info.json").read_text()),
    }
    (args.out / "results.json").write_text(
        json.dumps(results, separators=(",", ":"), allow_nan=False)
    )
    for f in sorted(args.out.glob("*.json")):
        print(f.name, f"{f.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
