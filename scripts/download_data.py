"""Verify the three AFID archives in data/archives/ and extract them to data/raw/.

The authors do not publish a stable direct download link, so the archives must be
downloaded by hand from https://www.aitex.es/afid/ and placed in data/archives/.
"""

import argparse
import sys
from pathlib import Path

from fabricdefect.dataset import ARCHIVE_DIR, RAW_DIR
from fabricdefect.download import extract_archives, verify_archives


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archives", type=Path, default=ARCHIVE_DIR)
    ap.add_argument("--out", type=Path, default=RAW_DIR)
    args = ap.parse_args()

    problems = verify_archives(args.archives)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        print("Download the archives from https://www.aitex.es/afid/ into", args.archives)
        return 1
    extract_archives(args.archives, args.out)
    print(f"Archives verified and extracted to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
