"""Cut strips into square patches, label them from masks, and rebuild score maps.

Strips are 256 px high, so patches span the full height and only slide along the width.
"""

from __future__ import annotations

import numpy as np

PATCH_SIZE = 256
STRIDE = 128  # 50% overlap
MIN_DEFECT_PX = 5  # see docs/patching_and_splits.md for why

CLEAN, DEFECTIVE, AMBIGUOUS = 0, 1, -1


def patch_starts(width: int, size: int = PATCH_SIZE, stride: int = STRIDE) -> list[int]:
    """Left edges of the patches. A final patch is aligned to the right edge so that
    every pixel is covered even when (width - size) is not a multiple of the stride."""
    if width < size:
        raise ValueError(f"strip width {width} is smaller than the patch size {size}")
    starts = list(range(0, width - size + 1, stride))
    if starts[-1] + size < width:
        starts.append(width - size)
    return starts


def extract_patches(img: np.ndarray, starts: list[int], size: int = PATCH_SIZE) -> np.ndarray:
    """Stack of patches, shape (n, size, size)."""
    if img.shape[0] != size:
        raise ValueError(f"strip height {img.shape[0]} must equal the patch size {size}")
    return np.stack([img[:, x : x + size] for x in starts])


def defect_pixels_per_patch(
    mask: np.ndarray, starts: list[int], size: int = PATCH_SIZE
) -> np.ndarray:
    """Number of defect pixels inside each patch."""
    return np.array([int(mask[:, x : x + size].sum()) for x in starts])


def label_patches(defect_px: np.ndarray, min_px: int = MIN_DEFECT_PX) -> np.ndarray:
    """DEFECTIVE if at least `min_px` defect pixels, CLEAN if none, otherwise AMBIGUOUS.

    Ambiguous patches contain a sliver of a defect (typically the edge of one that lies
    mostly in a neighbouring patch). They are excluded from patch-level training and
    scoring rather than being forced to a label either way.
    """
    labels = np.full(defect_px.shape, AMBIGUOUS, dtype=np.int8)
    labels[defect_px == 0] = CLEAN
    labels[defect_px >= min_px] = DEFECTIVE
    return labels


def assemble_score_map(
    scores: np.ndarray, starts: list[int], width: int, size: int = PATCH_SIZE
) -> np.ndarray:
    """Spread one score per patch back over the strip; overlapping patches are averaged.

    Returns a (size, width) float array.
    """
    acc = np.zeros((size, width))
    cnt = np.zeros((size, width))
    for s, x in zip(scores, starts, strict=True):
        acc[:, x : x + size] += s
        cnt[:, x : x + size] += 1
    if (cnt == 0).any():
        raise ValueError("patches do not cover the whole strip")
    return acc / cnt


def assemble_column_scores(
    scores: np.ndarray, starts: list[int], width: int, size: int = PATCH_SIZE
):
    """Like `assemble_score_map` but returns only the (width,) per-column scores.

    Patches span the full strip height, so every row of the score map is identical and the
    1-D form carries all the information at a fraction of the memory.
    """
    acc = np.zeros(width)
    cnt = np.zeros(width)
    for s, x in zip(scores, starts, strict=True):
        acc[x : x + size] += s
        cnt[x : x + size] += 1
    if (cnt == 0).any():
        raise ValueError("patches do not cover the whole strip")
    return acc / cnt
