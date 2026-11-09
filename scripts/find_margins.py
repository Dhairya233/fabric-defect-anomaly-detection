"""Flag patches that touch a blank margin and summarise how common margins are.

Writes
  data/cache/margin_patch.npy   one boolean per patch, in the order of data/patch_data.npz
  results/margin_summary.csv    the counts quoted in the README, site and video
"""

import csv
from pathlib import Path

import numpy as np

from fabricdefect.cli import require
from fabricdefect.data import load_image
from fabricdefect.dataset import MANIFEST_PATH
from fabricdefect.margins import margin_patch_flags, saturated_fraction
from fabricdefect.patchdata import CACHE_PATH, PatchData


def main() -> int:
    require(CACHE_PATH, "Run python scripts/extract_features.py first.")
    require(MANIFEST_PATH, "Run python scripts/prepare_data.py first.")
    pdat = PatchData.load(CACHE_PATH)
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    assert [r["image_id"] for r in rows] == pdat.image_ids.tolist()

    flags, fractions = [], []
    for i, r in enumerate(rows):
        img = load_image(r["path"])
        flags.append(margin_patch_flags(img, pdat.patch_x[pdat.patches_of(i)].tolist()))
        fractions.append(saturated_fraction(img))
    flags = np.concatenate(flags)
    frac = np.array(fractions)
    np.save(Path("data/cache/margin_patch.npy"), flags)

    lab = pdat.patch_label
    summary = [
        ("strips", len(rows)),
        ("strips_with_blank_columns", int((frac > 0).sum())),
        ("strips_at_least_20_percent_blank", int((frac >= 0.20).sum())),
        ("max_blank_fraction_of_a_strip", round(float(frac.max()), 3)),
        ("patches", len(flags)),
        ("patches_touching_a_margin", int(flags.sum())),
        ("defective_patches", int((lab == 1).sum())),
        ("defective_patches_touching_a_margin", int((flags & (lab == 1)).sum())),
    ]
    Path("results").mkdir(exist_ok=True)
    with open("results/margin_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        w.writerows(summary)
    for k, v in summary:
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
