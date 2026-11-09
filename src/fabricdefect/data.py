"""Loading strips and masks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import MASK_THRESHOLD


def load_image(path: str | Path) -> np.ndarray:
    """Load a strip as a 2-D uint8 array. Alpha channels (LA files) are discarded."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def load_mask(paths: Sequence[str | Path], shape: tuple[int, int]) -> np.ndarray:
    """Union of one or more mask files as a boolean array of the given (height, width).

    Masks are anti-aliased, so a pixel is defect if its value is > MASK_THRESHOLD.
    An empty `paths` gives an all-False mask (a defect-free strip).
    """
    out = np.zeros(shape, dtype=bool)
    for p in paths:
        m = load_image(p) > MASK_THRESHOLD
        if m.shape != shape:
            raise ValueError(f"mask {p} has shape {m.shape}, expected {shape}")
        out |= m
    return out
