"""Compute texture features for every patch and cache them in data/patch_data.npz.
About 4 s per strip on one CPU core (roughly 15 minutes in total). Finished strips are
cached in data/cache/strips/, so an interrupted run can simply be started again."""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

from fabricdefect.cli import require
from fabricdefect.dataset import MANIFEST_PATH
from fabricdefect.patchdata import CACHE_PATH, build_patch_data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--out", type=Path, default=CACHE_PATH)
    args = ap.parse_args()

    require(args.manifest, "Run python scripts/prepare_data.py first.")
    with open(args.manifest) as f:
        records = [
            SimpleNamespace(**{**r, "defect_code": int(r["defect_code"])})
            for r in csv.DictReader(f)
        ]

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"\r{done}/{total} strips", end="", file=sys.stderr, flush=True)

    data = build_patch_data(records, progress, strip_cache=Path("data/cache/strips"))
    data.save(args.out)
    print(f"\n{len(data.patch_x)} patches x {data.features.shape[1]} features -> {args.out}")
    print(f"extraction: {data.extract_seconds.sum():.0f} s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
