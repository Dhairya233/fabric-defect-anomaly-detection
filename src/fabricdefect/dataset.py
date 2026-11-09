"""Static facts about the AFID dataset: names, codes, exclusions, archive checksums."""

from __future__ import annotations

from pathlib import Path

ARCHIVES = {
    "Defect_images.7z": "2153ef691020ce0c6a0acaee2d1de76c75f0784a04b931706de7d5e1df7a1b63",
    "NODefect_images.7z": "4409ca615f069e0b92eaf1bd2058c8a37d9cb983f3aad205e7d69f30816f5664",
    "Mask_images.7z": "2d64486dbb5a2e1aa462b43c0f19e614af5aea6607c11d1a928e4ad9feb558d5",
}
# These SHA-256 values were computed from the archives as downloaded from the dataset
# page; the authors do not publish checksums.

DEFECT_NAMES = {
    0: "none",
    2: "broken end",
    6: "broken yarn",
    10: "broken pick",
    16: "weft curling",
    19: "fuzzyball",
    22: "cut selvage",
    23: "crease",
    25: "warp ball",
    27: "knots",
    29: "contamination",
    30: "nep",
    36: "weft crack",
}

# Files that do not match the dataset description (140 clean, 105 defective).
EXCLUDED = {
    "0018_00_01": "21st clean image for fabric 01; name breaks the nnnn_000_ff pattern",
    "0100_025_08": "fabric code 08 does not exist elsewhere; no mask",
}

MASK_THRESHOLD = 127  # mask pixels > this value count as defect

DATA_DIR = Path("data")
ARCHIVE_DIR = DATA_DIR / "archives"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
