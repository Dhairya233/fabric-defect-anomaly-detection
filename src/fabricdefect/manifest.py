"""Build the image manifest: one row per usable AFID image, with its mask file(s)."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import DEFECT_NAMES, EXCLUDED, MASK_THRESHOLD

_IMAGE_RE = re.compile(r"^(\d{4})_(\d{2,3})_(\d{2})$")
_MASK_RE = re.compile(r"^(\d{4}_\d{2,3}_\d{2})_mask\d?$")


@dataclass
class Record:
    image_id: str
    path: str
    kind: str  # "clean" or "defective"
    fabric: str
    defect_code: int
    defect_name: str
    width: int
    height: int
    mask_paths: str  # ';'-separated; empty for clean images
    mask_empty: bool  # True if the mask has no pixel above the threshold


def parse_image_id(stem: str) -> tuple[str, int, str] | None:
    """Split 'nnnn_ddd_ff' into (image number, defect code, fabric code)."""
    m = _IMAGE_RE.match(stem)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def mask_owner(stem: str) -> str | None:
    """Return the image id a mask file belongs to ('0044_019_04_mask1' -> '0044_019_04')."""
    m = _MASK_RE.match(stem)
    return m.group(1) if m else None


def _find_pngs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.png"))


def _mask_has_defect(mask_files: list[Path]) -> bool:
    for p in mask_files:
        with Image.open(p) as im:
            if (np.asarray(im.convert("L")) > MASK_THRESHOLD).any():
                return True
    return False


def build_manifest(raw_dir: Path) -> list[Record]:
    """Scan the extracted archives and return the manifest rows, sorted by image id."""
    masks: dict[str, list[Path]] = {}
    for p in _find_pngs(raw_dir / "Mask_images"):
        owner = mask_owner(p.stem)
        if owner:
            masks.setdefault(owner, []).append(p)

    records: list[Record] = []
    for folder, kind in (("Defect_images", "defective"), ("NODefect_images", "clean")):
        for p in _find_pngs(raw_dir / folder):
            if p.stem in EXCLUDED:
                continue
            parsed = parse_image_id(p.stem)
            if parsed is None:
                raise ValueError(f"unrecognised file name: {p.name}")
            _, code, fabric = parsed
            if (kind == "clean") != (code == 0):
                raise ValueError(f"defect code {code} does not match folder {folder}: {p.name}")
            if code not in DEFECT_NAMES:
                raise ValueError(f"unknown defect code {code}: {p.name}")
            with Image.open(p) as im:
                width, height = im.size
            mask_files = sorted(masks.get(p.stem, []))
            if kind == "defective" and not mask_files:
                raise ValueError(f"defective image without a mask: {p.name}")
            records.append(
                Record(
                    image_id=p.stem,
                    path=str(p),
                    kind=kind,
                    fabric=fabric,
                    defect_code=code,
                    defect_name=DEFECT_NAMES[code],
                    width=width,
                    height=height,
                    mask_paths=";".join(str(m) for m in mask_files),
                    mask_empty=bool(mask_files) and not _mask_has_defect(mask_files),
                )
            )
    return sorted(records, key=lambda r: r.image_id)


def write_manifest(records: list[Record], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[x.name for x in fields(Record)])
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def summarise(records: list[Record]) -> str:
    """Counts per fabric, as a small text table."""
    fabrics = sorted({r.fabric for r in records})
    lines = ["fabric  clean  defective"]
    for f in fabrics:
        clean = sum(r.fabric == f and r.kind == "clean" for r in records)
        bad = sum(r.fabric == f and r.kind == "defective" for r in records)
        lines.append(f"{f:>6}  {clean:>5}  {bad:>9}")
    lines.append(
        f" total  {sum(r.kind == 'clean' for r in records):>5}  "
        f"{sum(r.kind == 'defective' for r in records):>9}"
    )
    return "\n".join(lines)
