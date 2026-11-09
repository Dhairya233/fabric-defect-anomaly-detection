"""Scan data/raw/, apply the documented exclusions and write data/manifest.csv."""

import argparse
from pathlib import Path

from fabricdefect.cli import require
from fabricdefect.dataset import MANIFEST_PATH, RAW_DIR
from fabricdefect.manifest import build_manifest, summarise, write_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=MANIFEST_PATH)
    args = ap.parse_args()

    require(args.raw / "Defect_images", "Run python scripts/download_data.py first (see README).")
    records = build_manifest(args.raw)
    if not records:
        raise SystemExit(f"error: no images found under {args.raw}; run scripts/download_data.py")
    write_manifest(records, args.out)
    print(summarise(records))
    clean = sum(r.kind == "clean" for r in records)
    if (clean, len(records) - clean) != (140, 105):
        print(
            f"warning: expected 140 clean and 105 defective images, found {clean} and "
            f"{len(records) - clean}; the archives may be incomplete or different"
        )
    print(f"Wrote {len(records)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
