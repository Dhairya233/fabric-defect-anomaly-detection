"""One table holding every patch of every strip: features, labels, positions.

Patches are stored image by image, so the patches of image i are
`patch_offsets[i]:patch_offsets[i + 1]`. Per-column defect-pixel counts are kept for each
strip so localisation can be scored exactly without storing full masks.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import load_image, load_mask
from .features import feature_names, strip_features
from .patches import defect_pixels_per_patch, label_patches, patch_starts

CACHE_PATH = Path("data/patch_data.npz")


@dataclass
class PatchData:
    image_ids: np.ndarray
    fabrics: np.ndarray
    kinds: np.ndarray
    codes: np.ndarray
    widths: np.ndarray
    patch_offsets: np.ndarray
    col_offsets: np.ndarray
    col_defect: np.ndarray  # defect pixels per column, concatenated over strips
    patch_x: np.ndarray
    patch_px: np.ndarray
    patch_label: np.ndarray
    features: np.ndarray
    raw_mean: np.ndarray
    raw_std: np.ndarray
    extract_seconds: np.ndarray  # per strip
    feature_names: np.ndarray

    @property
    def n_images(self) -> int:
        return len(self.image_ids)

    @property
    def patch_image(self) -> np.ndarray:
        """Index of the owning image for every patch."""
        return np.repeat(np.arange(self.n_images), np.diff(self.patch_offsets))

    def patches_of(self, i: int) -> slice:
        return slice(int(self.patch_offsets[i]), int(self.patch_offsets[i + 1]))

    def columns_of(self, i: int) -> slice:
        return slice(int(self.col_offsets[i]), int(self.col_offsets[i + 1]))

    def image_index(self, image_ids: Sequence[str]) -> np.ndarray:
        lookup = {k: j for j, k in enumerate(self.image_ids)}
        return np.array([lookup[k] for k in image_ids], dtype=int)

    def patches_for_images(self, image_idx: Sequence[int]) -> np.ndarray:
        if len(image_idx) == 0:
            return np.zeros(0, dtype=int)
        return np.concatenate(
            [np.arange(self.patch_offsets[i], self.patch_offsets[i + 1]) for i in image_idx]
        )

    def save(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path = CACHE_PATH) -> PatchData:
        with np.load(path, allow_pickle=False) as z:
            return cls(**{k: z[k] for k in z.files})


STRIP_KEYS = "patch_x patch_px patch_label features raw_mean raw_std col_defect extract_seconds"


def compute_strip(record) -> dict[str, np.ndarray]:
    """All per-strip arrays for one manifest record."""
    img = load_image(record.path)
    h, w = img.shape
    starts = patch_starts(w)
    mask = load_mask([p for p in record.mask_paths.split(";") if p], (h, w))
    px = defect_pixels_per_patch(mask, starts)
    t0 = time.perf_counter()
    feats, rm, rs = strip_features(img, starts)
    return {
        "patch_x": np.array(starts),
        "patch_px": px,
        "patch_label": label_patches(px),
        "features": feats,
        "raw_mean": rm,
        "raw_std": rs,
        "col_defect": mask.sum(axis=0).astype(np.int16),
        "extract_seconds": np.array(time.perf_counter() - t0),
    }


def build_patch_data(
    records: Sequence,
    progress: Callable[[int, int], None] | None = None,
    strip_cache: Path | None = None,
) -> PatchData:
    """records: manifest Record objects (see fabricdefect.manifest).

    With `strip_cache`, each finished strip is saved there and reused on the next run, so
    an interrupted extraction resumes where it stopped.
    """
    strips = []
    for n, r in enumerate(records):
        cached = strip_cache / f"{r.image_id}.npz" if strip_cache else None
        if cached is not None and cached.exists():
            with np.load(cached) as z:
                strips.append({k: z[k] for k in z.files})
        else:
            st = compute_strip(r)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cached, **st)
            strips.append(st)
        if progress:
            progress(n + 1, len(records))
    widths = np.array([len(st["col_defect"]) for st in strips])
    counts = [len(st["patch_x"]) for st in strips]
    cat = {k: np.concatenate([st[k] for st in strips]) for k in STRIP_KEYS.split()[:6]}
    return PatchData(
        image_ids=np.array([r.image_id for r in records]),
        fabrics=np.array([r.fabric for r in records]),
        kinds=np.array([r.kind for r in records]),
        codes=np.array([r.defect_code for r in records]),
        widths=widths,
        patch_offsets=np.r_[0, np.cumsum(counts)],
        col_offsets=np.r_[0, np.cumsum(widths)],
        col_defect=np.concatenate([st["col_defect"] for st in strips]),
        extract_seconds=np.array([float(st["extract_seconds"]) for st in strips]),
        feature_names=np.array(feature_names()),
        **cat,
    )
