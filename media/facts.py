"""Every number that appears in the video, read from the regenerated result tables.

Nothing here is typed in by hand: change the results and the media changes with them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def load() -> dict[str, str]:
    results = json.loads((ROOT / "site/data/results.json").read_text())
    masked = results["runs"]["masked"]
    rf = masked["table"]["rf"]
    counts = masked["counts"]
    grouped_ap, leaky_ap = rf["grouped"]["patch_ap"][0], rf["leaky"]["patch_ap"][0]
    gap, gap_lo, gap_hi = masked["gaps"]["rf"]["leaky minus grouped"]["patch_ap"]
    unseen, unseen_lo, unseen_hi = results["artefact_free_subset"]["lofo:rf"]
    margin = {
        r["quantity"]: float(r["value"])
        for r in csv.DictReader(open(ROOT / "results/margin_summary.csv"))
    }
    art = {str(r["fabric"]).zfill(2): r for r in results["artefact_check"]}
    ratio = leaky_ap / grouped_ap
    return {
        "n_strips": str(counts["images"]),
        "n_defective": str(counts["defective_images"]),
        "n_fabrics": str(len(art)),
        "rf_ap_grouped": _fmt(grouped_ap),
        "rf_ap_leaky": _fmt(leaky_ap),
        "ratio": _fmt(ratio, 1),
        "gap": _fmt(gap),
        "gap_lo": _fmt(gap_lo),
        "gap_hi": _fmt(gap_hi),
        "unseen": _fmt(unseen),
        "unseen_lo": _fmt(unseen_lo),
        "unseen_hi": _fmt(unseen_hi),
        "margin_strips": str(int(margin["strips_with_blank_columns"])),
        "bright_auc": _fmt(art["01"]["auroc_brightness"]),
        "_rf_ap_grouped_raw": grouped_ap,
        "_rf_ap_leaky_raw": leaky_ap,
    }


if __name__ == "__main__":
    for k, v in load().items():
        print(f"{k:18s} {v}")
