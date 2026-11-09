"""Experiments that test specific criticisms of the evaluation (see docs/review.md).

  python scripts/review_checks.py matched     strip scores with equal numbers of patches
  python scripts/review_checks.py seeds       variance from the random fold assignment
  python scripts/review_checks.py fabrics     intervals that resample fabrics, not strips
  python scripts/review_checks.py reproduce   rerun one method and compare with the cache
  python scripts/review_checks.py duplicates  near-duplicate strips across the dataset

Outputs go to results/review/. Needs data/cache/ from the experiment runs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fabricdefect.cli import require
from fabricdefect.evaluation import (
    MIN_DEFECTIVE_FOR_FABRIC_AVERAGE,
    _auroc,
    count_matched_image_scores,
    metrics,
    prepare,
)
from fabricdefect.experiments import build_folds, run_method
from fabricdefect.models import RandomForest, method_factories
from fabricdefect.patchdata import PatchData

OUT = Path("results/review")
MASK = Path("data/cache/margin_patch.npy")
PRED = Path("data/cache/predictions_margin.npz")


def macro(y, s, fab, w):
    out = []
    for f in np.unique(fab):
        m = fab == f
        if (w * (y & m)).sum() >= MIN_DEFECTIVE_FOR_FABRIC_AVERAGE and (w * (~y & m)).sum() > 0:
            out.append(_auroc(y[m], s[m], w[m]))
    return float(np.nanmean(out))


def cmd_matched(n_boot: int = 500) -> None:
    pdat, valid = PatchData.load(), ~np.load(MASK)
    z = np.load(PRED)
    y, fab = pdat.kinds == "defective", pdat.fabrics
    rng = np.random.default_rng(0)
    draws = [np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)) for _ in range(n_boot)]
    rows = []
    for key in z.files:
        split, method, what = key.split("|")
        if what != "scores" or method == "M0 always clean":
            continue
        sc = z[key]
        orig = np.array([sc[pdat.patches_of(i)].max() for i in range(pdat.n_images)])
        matched = count_matched_image_scores(pdat, sc, valid)
        for name, s in (("as scored", orig), ("count-matched", matched)):
            ones = np.ones(len(y))
            boot = np.array([macro(y, s, fab, w.astype(float)) for w in draws])
            f01 = _auroc(y[fab == "01"], s[fab == "01"], np.ones((fab == "01").sum()))
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "image_score": name,
                    "macro_fabric_auroc": macro(y, s, fab, ones),
                    "ci_low": np.nanquantile(boot, 0.025),
                    "ci_high": np.nanquantile(boot, 0.975),
                    "fabric_01_auroc": f01,
                }
            )
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "count_matched.csv", index=False, float_format="%.3f")
    print(df.round(2).to_string(index=False))


def cmd_seeds(seeds=(1, 2, 3, 4)) -> None:
    pdat, valid = PatchData.load(), ~np.load(MASK)
    wanted = {"M1 random forest", "M1 SVM (RBF)", "M2 Mahalanobis (per fabric)"}
    rows = []
    for seed in (0, *seeds):
        folds = build_folds(pdat, seed=seed)["grouped"]
        for name, factory in method_factories():
            if name not in wanted:
                continue
            run = run_method(pdat, folds, factory, valid=valid)
            m = metrics(prepare(pdat, run.scores, run.thr, valid), np.ones(pdat.n_images))
            rows.append(
                {
                    "seed": seed,
                    "method": name,
                    "image_macro_fabric_auroc": m["image_macro_fabric_auroc"],
                    "image_auroc": m["image_auroc"],
                    "patch_ap": m["patch_ap"],
                }
            )
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "fold_assignment_seeds.csv", index=False, float_format="%.3f")
    print(
        df.groupby("method")[["image_macro_fabric_auroc", "image_auroc", "patch_ap"]]
        .agg(["mean", "std", "min", "max"])
        .round(3)
        .to_string()
    )


def cmd_fabrics(n_boot: int = 10000) -> None:
    f = pd.read_csv("results/margin_masked/by_fabric.csv")
    f = f[f.n_defective >= MIN_DEFECTIVE_FOR_FABRIC_AVERAGE]
    rng = np.random.default_rng(0)
    rows = []
    for method, g in f.groupby("method"):
        wide = g.pivot(index="fabric", columns="split", values="auroc")
        if "lofo" not in wide.columns:  # per-fabric models cannot run on an unseen fabric
            continue
        wide = wide.dropna(subset=["grouped", "lofo"])
        if len(wide) < 3:
            continue
        for label, v in (
            ("grouped", wide.grouped.values),
            ("unseen fabric", wide.lofo.values),
            ("grouped minus unseen", (wide.grouped - wide.lofo).values),
        ):
            boots = np.array([rng.choice(v, len(v)).mean() for _ in range(n_boot)])
            rows.append(
                {
                    "method": method,
                    "quantity": label,
                    "n_fabrics": len(v),
                    "mean_per_fabric_auroc": v.mean(),
                    "ci_low": np.quantile(boots, 0.025),
                    "ci_high": np.quantile(boots, 0.975),
                    "fabrics_where_grouped_is_higher": int((wide.grouped > wide.lofo).sum()),
                }
            )
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "fabric_level_intervals.csv", index=False, float_format="%.3f")
    print(df.round(2).to_string(index=False))


def cmd_reproduce() -> None:
    pdat = PatchData.load()
    z = np.load("data/cache/predictions.npz")
    run = run_method(pdat, build_folds(pdat)["grouped"], RandomForest)
    same = np.allclose(run.scores, z["grouped|M1 random forest|scores"]) and np.allclose(
        run.thr, z["grouped|M1 random forest|thr"]
    )
    print("random forest, grouped split, as-is: rerun matches the cached predictions:", same)
    print(
        "max abs difference:",
        float(np.abs(run.scores - z["grouped|M1 random forest|scores"]).max()),
    )


def cmd_duplicates() -> None:
    import csv

    from fabricdefect.data import load_image
    from fabricdefect.features import robust_normalise
    from fabricdefect.margins import margin_columns

    with open("data/manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    W = 4096
    spectra, fabrics = [], []
    for r in rows:
        img = load_image(r["path"])
        z = robust_normalise(img).astype(np.float32)
        z = z.reshape(64, 4, z.shape[1]).mean(axis=1)  # 4x row averaging, keeps columns
        keep = ~margin_columns(img)
        z[:, ~keep] = 0
        z[:, keep] -= z[:, keep].mean()
        pad = np.zeros((64, W), np.float32)
        pad[:, : z.shape[1]] = z
        spectra.append(np.fft.rfft2(pad))
        spectra[-1] = (spectra[-1], float(np.linalg.norm(pad)))
        fabrics.append(r["fabric"])
    n = len(rows)

    def ncc(i, j):
        (a, na), (b, nb) = spectra[i], spectra[j]
        c = np.fft.irfft2(a * np.conj(b), s=(64, W))
        return float(c.max() / (na * nb + 1e-9))

    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if fabrics[i] == fabrics[j]:
                within.append((ncc(i, j), fabrics[i], rows[i]["image_id"], rows[j]["image_id"]))
    rng = np.random.default_rng(0)
    while len(between) < 2000:
        i, j = rng.integers(0, n, 2)
        if fabrics[i] != fabrics[j]:
            between.append(ncc(i, j))
    df = pd.DataFrame(within, columns=["ncc", "fabric", "strip_a", "strip_b"])
    # A regular weave makes every pair of its strips correlate; a duplicated crop would be an
    # outlier among the pairs of its own fabric. Robust z = distance from the fabric median in
    # units of the median absolute deviation.
    rz = []
    for _, g in df.groupby("fabric"):
        med = g.ncc.median()
        mad = (g.ncc - med).abs().median() * 1.4826 + 1e-9
        rz.append((g.ncc - med) / mad)
    df["robust_z_within_fabric"] = pd.concat(rz)
    per_fabric = df.groupby("fabric").ncc.agg(
        pairs="size", median="median", p99=lambda x: x.quantile(0.99), max="max"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    per_fabric.to_csv(OUT / "near_duplicates_by_fabric.csv", float_format="%.3f")
    df.sort_values("robust_z_within_fabric", ascending=False).head(10).to_csv(
        OUT / "near_duplicates_top_pairs.csv", index=False, float_format="%.3f"
    )
    print(
        f"different-fabric pairs (null): median {np.median(between):.2f}, max {np.max(between):.2f}"
    )
    print(per_fabric.round(2).to_string())
    print("pairs furthest above their own fabric's typical correlation:")
    print(
        df.sort_values("robust_z_within_fabric", ascending=False)
        .head(6)
        .round(2)
        .to_string(index=False)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("check", choices=["matched", "seeds", "fabrics", "reproduce", "duplicates"])
    args = ap.parse_args()
    for need in ("data/patch_data.npz", "data/manifest.csv"):
        require(need, "Run the earlier steps in the README first.")
    if args.check in ("matched", "seeds"):
        require(MASK, "Run python scripts/find_margins.py first.")
    if args.check in ("matched", "reproduce"):
        require(
            PRED if args.check == "matched" else "data/cache/predictions.npz",
            "Run python scripts/run_experiments.py first (see README).",
        )
    {
        "matched": cmd_matched,
        "seeds": cmd_seeds,
        "fabrics": cmd_fabrics,
        "reproduce": cmd_reproduce,
        "duplicates": cmd_duplicates,
    }[args.check]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
