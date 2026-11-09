"""Do clean and defective strips of the same fabric differ in ways unrelated to defects?

Writes
  results/artefact_check.csv       per fabric: brightness of clean vs defective strips and
                                   how well brightness / contrast alone separate them
  results/artefact_free_subset.csv fabrics where they do NOT differ: mean of per-fabric
                                   image AUROC for each method, with bootstrap intervals

A defect covers about 0.03% of a strip, so it cannot move a strip's mean brightness. If
brightness separates the classes within a fabric, the strips were captured under
different conditions, and any score that picks this up is not detecting defects.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fabricdefect.cli import require
from fabricdefect.evaluation import _auroc, prepare
from fabricdefect.patchdata import PatchData

# Rule: the clean and defective strips of the fabric differ in mean brightness by less than
# one clean-strip standard deviation. That selects exactly 03, 04 and 06 (fabric 00 differs
# by 3.2 sd, 02 by 5.8, 01 by 15, 05 by 15). The rule was written after looking at these
# numbers, so treat the subset as a sensitivity analysis, not a pre-registered test.
SUBSET = ("03", "04", "06")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--predictions", type=Path, default=Path("data/cache/predictions_margin.npz"))
    ap.add_argument("--margins", type=Path, default=Path("data/cache/margin_patch.npy"))
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    require("data/patch_data.npz", "Run python scripts/extract_features.py first.")
    require(args.margins, "Run python scripts/find_margins.py first.")
    require(
        args.predictions,
        "Run the margin-masked experiment first (see README).",
    )
    p = PatchData.load()
    valid = ~np.load(args.margins)
    rows = []
    for i in range(p.n_images):
        sl = p.patches_of(i)
        v = valid[sl]
        rows.append(
            (
                p.fabrics[i],
                p.kinds[i] == "defective",
                p.raw_mean[sl][v].mean(),
                p.raw_std[sl][v].mean(),
            )
        )
    d = pd.DataFrame(rows, columns=["fabric", "defective", "mean", "std"])
    out = []
    for f, g in d.groupby("fabric"):
        c, b = g[~g.defective], g[g.defective]
        both = g.defective.nunique() == 2
        out.append(
            {
                "fabric": f,
                "n_clean": len(c),
                "n_defective": len(b),
                "clean_mean_brightness": c["mean"].mean(),
                "clean_sd_across_strips": c["mean"].std(),
                "defective_mean_brightness": b["mean"].mean(),
                "defective_sd_across_strips": b["mean"].std(),
                "auroc_brightness": roc_auc_score(g.defective, g["mean"]) if both else np.nan,
                "auroc_contrast": roc_auc_score(g.defective, g["std"]) if both else np.nan,
                "in_artefact_free_subset": f in SUBSET,
            }
        )
    args.out.mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv(args.out / "artefact_check.csv", index=False, float_format="%.3f")

    z = np.load(args.predictions)
    fab = p.fabrics
    sub = np.isin(fab, SUBSET)
    rng = np.random.default_rng(0)
    draws = [
        np.bincount(rng.integers(0, p.n_images, p.n_images), minlength=p.n_images)
        for _ in range(args.boot)
    ]
    res = []
    methods = sorted({k.split("|")[1] for k in z.files if k.startswith("grouped|")})
    for split in ("grouped", "lofo", "leaky"):
        for m in methods:
            if f"{split}|{m}|scores" not in z.files:
                continue
            prep = prepare(p, z[f"{split}|{m}|scores"], z[f"{split}|{m}|thr"], valid)
            y, s = prep.img_defective, prep.img_score

            def macro(w, y=y, s=s):
                a = [_auroc(y[fab == f], s[fab == f], w[fab == f]) for f in SUBSET]
                return float(np.nanmean(a))

            point = macro(np.ones(p.n_images))
            boot = np.array([macro(w.astype(float)) for w in draws])
            ok = boot[~np.isnan(boot)]
            res.append(
                {
                    "split": split,
                    "method": m,
                    "fabrics": "+".join(SUBSET),
                    "n_defective_images": int((y & sub).sum()),
                    "mean_per_fabric_image_auroc": point,
                    "ci_low": np.quantile(ok, 0.025),
                    "ci_high": np.quantile(ok, 0.975),
                }
            )
    pd.DataFrame(res).to_csv(
        args.out / "artefact_free_subset.csv", index=False, float_format="%.3f"
    )
    print(pd.DataFrame(out).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
