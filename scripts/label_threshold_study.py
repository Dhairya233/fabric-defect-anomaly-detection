"""How does the minimum-defect-pixels rule change patch labels? Writes
results/patch_label_threshold.csv (used to justify MIN_DEFECT_PX)."""

import argparse
import csv
from pathlib import Path

import numpy as np

from fabricdefect.data import load_mask
from fabricdefect.dataset import MANIFEST_PATH
from fabricdefect.patches import CLEAN, DEFECTIVE, label_patches, patch_starts

THRESHOLDS = [1, 5, 10, 25, 50, 100, 500, 2000]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--out", type=Path, default=Path("results/patch_label_threshold.csv"))
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["kind"] == "defective"]

    counts = {}
    for r in rows:
        w, h = int(r["width"]), int(r["height"])
        mask = load_mask(r["mask_paths"].split(";"), (h, w))
        starts = patch_starts(w)
        px = np.array([int(mask[:, x : x + 256].sum()) for x in starts])
        counts[r["image_id"]] = px

    n_images = len(counts)
    out = []
    for t in THRESHOLDS:
        pos = amb = clean = missed = 0
        for px in counts.values():
            lab = label_patches(px, t)
            pos += int((lab == DEFECTIVE).sum())
            clean += int((lab == CLEAN).sum())
            amb += int((lab == -1).sum())
            missed += int(not (lab == DEFECTIVE).any())
        out.append(
            {
                "min_defect_px": t,
                "defective_images": n_images,
                "images_with_no_positive_patch": missed,
                "positive_patches": pos,
                "ambiguous_patches": amb,
                "clean_patches_in_defective_images": clean,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for row in out:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
