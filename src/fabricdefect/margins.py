"""Find the blank margins that some AFID strips have next to the fabric.

In 204 of the 245 strips some columns are saturated white (the camera sees past the
fabric). How much margin a strip has differs between clean and defective strips of the
same fabric, so a model can pick up "has a margin" instead of "has a defect". Patches
that touch a margin (plus a safety band for the dark edge next to it) are flagged so that
a sensitivity analysis can leave them out.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from .patches import PATCH_SIZE

SATURATED = 250  # a column whose mean intensity is at least this is blank
DILATION = 64  # px added on each side to cover the dark edge band


def margin_columns(img: np.ndarray, dilation: int = DILATION) -> np.ndarray:
    """Boolean per column: blank, or within `dilation` px of a blank column."""
    blank = img.astype(np.float64).mean(axis=0) >= SATURATED
    return binary_dilation(blank, iterations=dilation) if blank.any() else blank


def margin_patch_flags(
    img: np.ndarray, starts: list[int], size: int = PATCH_SIZE, dilation: int = DILATION
) -> np.ndarray:
    """True for every patch whose window touches a margin column."""
    cols = margin_columns(img, dilation)
    return np.array([bool(cols[x : x + size].any()) for x in starts])


def saturated_fraction(img: np.ndarray) -> float:
    """Share of columns that are blank (before any safety band is added)."""
    return float((img.astype(np.float64).mean(axis=0) >= SATURATED).mean())
