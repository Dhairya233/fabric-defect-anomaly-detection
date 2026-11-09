"""Draw the result figures from results/*.csv.

results/figures/   charts of our own numbers (safe to commit)
results/local/     failure gallery with dataset images (git-ignored: the dataset has no
                   stated licence, so its images are not redistributed)
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fabricdefect.cli import require
from fabricdefect.data import load_image, load_mask
from fabricdefect.patchdata import PatchData
from fabricdefect.patches import assemble_column_scores

SPLIT_STYLE = {
    "grouped": ("#0072B2", "grouped by image"),
    "lofo": ("#D55E00", "unseen fabric"),
    "leaky": ("#999999", "leaky patch split"),
}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def short(m: str) -> str:
    return m.replace("(global)", "").replace("(RBF)", "").replace("M0 ", "").strip()


def metric_table(overall: pd.DataFrame, metric: str) -> pd.DataFrame:
    return overall[overall["metric"] == metric].set_index(["method", "split"])


def leakage_figure(overall: pd.DataFrame, out: Path) -> None:
    methods = [m for m in overall["method"].unique() if not m.startswith("M0 always")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, metric, title in (
        (axes[0], "image_auroc", "Is the strip defective? (image AUROC)"),
        (axes[1], "patch_ap", "Which patches? (patch average precision)"),
    ):
        t = metric_table(overall, metric)
        y = np.arange(len(methods))[::-1]
        for j, split in enumerate(("grouped", "lofo", "leaky")):
            colour, label = SPLIT_STYLE[split]
            for yi, m in zip(y, methods, strict=True):
                if (m, split) not in t.index:
                    continue
                r = t.loc[(m, split)]
                ax.errorbar(
                    r["value"],
                    yi + (1 - j) * 0.22,
                    xerr=[[r["value"] - r["ci_low"]], [r["ci_high"] - r["value"]]],
                    fmt="o",
                    color=colour,
                    ms=4,
                    capsize=2,
                    label=label if yi == y[0] else None,
                )
        ax.set_yticks(y, [short(m) for m in methods])
        ax.set_title(title, loc="left", fontsize=10)
        ax.axvline(0.5 if metric.endswith("auroc") else 0, color="#cccccc", lw=0.8, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("The same models, three ways of splitting the data (95% intervals over images)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fabric_figure(by_fabric: pd.DataFrame, out: Path) -> None:
    d_all = by_fabric[~by_fabric["method"].str.startswith("M0 al")]
    methods = list(dict.fromkeys(d_all["method"]))
    fabrics = sorted(d_all["fabric"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, split in zip(axes, ("grouped", "lofo"), strict=True):
        d = d_all[d_all["split"] == split]
        grid = d.pivot(index="method", columns="fabric", values="auroc").reindex(
            index=methods, columns=fabrics
        )
        reliable = d.pivot(index="method", columns="fabric", values="reliable").reindex(
            index=methods, columns=fabrics
        )
        im = ax.imshow(grid.values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(fabrics)), [f"{int(f):02d}" for f in fabrics])
        ax.set_yticks(range(len(methods)), [short(m) for m in methods])
        for i in range(len(methods)):
            for j in range(len(fabrics)):
                v = grid.values[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#888888")
                    continue
                mark = "" if reliable.values[i, j] == True else "*"  # noqa: E712
                ax.text(
                    j,
                    i,
                    f"{v:.2f}{mark}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v < 0.6 else "black",
                )
        ax.set_title(SPLIT_STYLE[split][1], loc="left", fontsize=10)
        ax.set_xlabel("fabric")
    fig.colorbar(im, ax=axes, shrink=0.8, label="image AUROC within the fabric (0.5 = chance)")
    fig.text(0.01, 0.01, "* fewer than 5 defective images: not reliable", fontsize=8)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def defect_type_figure(by_type: pd.DataFrame, overall: pd.DataFrame, out: Path) -> None:
    t = metric_table(overall, "image_auroc")
    g = [
        m
        for m in overall["method"].unique()
        if m.startswith(("M1", "M2")) and (m, "grouped") in t.index
    ]
    best = max(g, key=lambda m: t.loc[(m, "grouped"), "value"])
    d = by_type[(by_type["split"] == "grouped") & (by_type["method"] == best)].sort_values(
        "n_defective", ascending=False
    )
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(d))
    ax.bar(x, d["recall"], color=np.where(d["reliable"], "#0072B2", "#bbbbbb"))
    ax.errorbar(
        x,
        d["recall"],
        yerr=[d["recall"] - d["recall_lo"], d["recall_hi"] - d["recall"]],
        fmt="none",
        ecolor="black",
        lw=0.8,
        capsize=2,
    )
    ax.set_xticks(
        x,
        [f"{n}\n(n={k})" for n, k in zip(d["defect"], d["n_defective"], strict=True)],
        fontsize=7,
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("share of defective strips flagged")
    ax.set_title(
        f"Recall by defect type: {short(best)}, grouped split\n"
        "method picked by its test AUROC, so descriptive only; grey = fewer than 5 examples",
        loc="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return best


def failure_gallery(
    best: str, pdat: PatchData, manifest: Path, out: Path, predictions: Path, n: int = 3
) -> None:
    with open(manifest) as f:
        rec = {r["image_id"]: r for r in csv.DictReader(f)}
    z = np.load(predictions)
    scores, thr = z[f"grouped|{best}|scores"], z[f"grouped|{best}|thr"]
    margin, cols = [], []
    for i in range(pdat.n_images):
        sl = pdat.patches_of(i)
        starts = pdat.patch_x[sl].tolist()
        w = int(pdat.widths[i])
        cols.append(assemble_column_scores(scores[sl] - thr[sl], starts, w))
        margin.append(float((scores[sl] - thr[sl]).max()))
    margin = np.array(margin)
    bad = pdat.kinds == "defective"
    fn = np.flatnonzero(bad & (margin < 0))
    fp = np.flatnonzero(~bad & (margin >= 0))
    fn = fn[np.argsort(margin[fn])][:n]
    fp = fp[np.argsort(-margin[fp])][:n]
    picks = [(i, "missed defect") for i in fn] + [(i, "false alarm") for i in fp]
    fig, axes = plt.subplots(
        len(picks),
        2,
        figsize=(12, 1.6 * len(picks)),
        gridspec_kw={"width_ratios": [4, 1]},
        squeeze=False,
    )
    for (i, kind), (ax, ax2) in zip(picks, axes, strict=True):
        r = rec[pdat.image_ids[i]]
        img = load_image(r["path"])
        ax.imshow(img, cmap="gray", aspect="auto")
        mask = load_mask([p for p in r["mask_paths"].split(";") if p], img.shape)
        if mask.any():
            ax.contour(mask, levels=[0.5], colors="#ff3b30", linewidths=0.8)
        ax.set_title(
            f"{pdat.image_ids[i]} fabric {pdat.fabrics[i]}: {kind}", fontsize=8, loc="left"
        )
        ax.set_yticks([])
        ax2.plot(cols[i], lw=0.8, color="#0072B2")
        ax2.axhline(0, color="#D55E00", lw=0.8)
        ax2.set_yticks([])
    fig.suptitle(
        f"{short(best)}: worst misses and false alarms (red outline = ground truth)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--predictions", type=Path, default=Path("data/cache/predictions.npz"))
    args = ap.parse_args()
    figs = args.results / "figures"
    local = args.results / "local"
    figs.mkdir(exist_ok=True)
    local.mkdir(exist_ok=True)

    require(args.results / "metrics_overall.csv", "Run python scripts/run_experiments.py first.")
    require("data/patch_data.npz", "Run python scripts/extract_features.py first.")
    require(args.predictions, "Run the experiment script that matches --results first.")
    overall = pd.read_csv(args.results / "metrics_overall.csv")
    overall = overall[overall["metric"] != "not applicable"]
    leakage_figure(overall, figs / "leakage.png")
    fabric_figure(pd.read_csv(args.results / "by_fabric.csv"), figs / "by_fabric.png")
    best = defect_type_figure(
        pd.read_csv(args.results / "by_defect_type.csv"), overall, figs / "by_defect_type.png"
    )
    failure_gallery(
        best,
        PatchData.load(),
        Path("data/manifest.csv"),
        local / "failure_gallery.png",
        args.predictions,
    )
    print("figures written to", figs, "and", local)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
