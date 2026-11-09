"""Train / validation / test splits.

Everything is split by *whole image* except `leaky_patch_kfold`, which exists only to
measure how much a careless patch-level split inflates results. Never use it to report a
headline number.

Roles of the three sides:
  train  fit models (anomaly models use only the defect-free images in here)
  val    choose thresholds and hyper-parameters
  test   evaluated once per configuration, never used for any choice
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

SEED = 0


class Item(Protocol):
    image_id: str
    fabric: str
    kind: str  # "clean" or "defective"


@dataclass(frozen=True)
class Split:
    name: str  # "grouped" or "lofo"
    fold: str  # fold number, or the held-out fabric for lofo
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def assert_disjoint(self) -> None:
        a, b, c = set(self.train), set(self.val), set(self.test)
        if a & b or a & c or b & c:
            raise AssertionError(f"image(s) shared between sides in {self.name} fold {self.fold}")


@dataclass(frozen=True)
class PatchSplit:
    """Index arrays into a flat list of patches."""

    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _strata(items: Sequence[Item]) -> dict[tuple[str, str], list[Item]]:
    out: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for it in sorted(items, key=lambda i: i.image_id):
        out[(it.fabric, it.kind)].append(it)
    return dict(sorted(out.items()))


def _carve_val(
    rest: Sequence[Item], val_frac: float, rng: np.random.Generator
) -> tuple[list[str], list[str]]:
    """Take about val_frac of each (fabric, kind) group, by image, as validation."""
    train, val = [], []
    for members in _strata(rest).values():
        order = rng.permutation(len(members))
        n_val = int(round(val_frac * len(members))) if len(members) >= 2 else 0
        for rank, idx in enumerate(order):
            (val if rank < n_val else train).append(members[idx].image_id)
    return sorted(train), sorted(val)


def grouped_kfold(
    items: Sequence[Item], k: int = 5, val_frac: float = 0.2, seed: int = SEED
) -> list[Split]:
    """k-fold cross-validation by whole image, stratified by (fabric, clean/defective)."""
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    offset = 0
    for members in _strata(items).values():
        for j, idx in enumerate(rng.permutation(len(members))):
            fold_of[members[idx].image_id] = (offset + j) % k
        offset += len(members)

    splits = []
    for f in range(k):
        test = sorted(i for i, g in fold_of.items() if g == f)
        rest = [it for it in items if fold_of[it.image_id] != f]
        tr, va = _carve_val(rest, val_frac, np.random.default_rng([seed, f]))
        s = Split("grouped", str(f), tuple(tr), tuple(va), tuple(test))
        s.assert_disjoint()
        splits.append(s)
    return splits


def leave_one_fabric_out(
    items: Sequence[Item], val_frac: float = 0.2, seed: int = SEED
) -> list[Split]:
    """One split per fabric: that fabric is the whole test side and appears nowhere else.

    Validation images come from the *other* fabrics, so thresholds are tuned on fabrics
    the model has seen, which is what a deployment without target-fabric data would do.
    """
    splits = []
    for n, fabric in enumerate(sorted({it.fabric for it in items})):
        test = sorted(it.image_id for it in items if it.fabric == fabric)
        rest = [it for it in items if it.fabric != fabric]
        tr, va = _carve_val(rest, val_frac, np.random.default_rng([seed, n]))
        s = Split("lofo", fabric, tuple(tr), tuple(va), tuple(test))
        s.assert_disjoint()
        splits.append(s)
    return splits


def leaky_patch_kfold(
    patch_image_ids: Sequence[str], k: int = 5, val_frac: float = 0.2, seed: int = SEED
) -> list[PatchSplit]:
    """DELIBERATELY LEAKY: random k-fold over patches, ignoring which image they came from.

    Overlapping patches of the same strip land on both sides, so a model can be scored on
    texture it has effectively already seen. Used only to measure the inflation.
    """
    n = len(patch_image_ids)
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(n) % k
    out = []
    for f in range(k):
        test = np.flatnonzero(fold_of == f)
        rest = np.flatnonzero(fold_of != f)
        rest = np.random.default_rng([seed, f]).permutation(rest)
        n_val = int(round(val_frac * len(rest)))
        out.append(PatchSplit(f, np.sort(rest[n_val:]), np.sort(rest[:n_val]), test))
    return out


def images_shared(patch_image_ids: Sequence[str], a: np.ndarray, b: np.ndarray) -> set[str]:
    """Images that have patches on both sides. Empty for any proper grouped split."""
    ids = np.asarray(patch_image_ids)
    return set(ids[a]) & set(ids[b])
