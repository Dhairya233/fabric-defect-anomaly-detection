import inspect
from collections import Counter
from types import SimpleNamespace as Item

import numpy as np
import pytest

from fabricdefect.patches import patch_starts
from fabricdefect.splits import (
    grouped_kfold,
    images_shared,
    leaky_patch_kfold,
    leave_one_fabric_out,
)
from fabricdefect.thresholds import best_f1_threshold


@pytest.fixture
def items():
    out = []
    n = 0
    for fabric, n_clean, n_bad in [("00", 20, 7), ("01", 20, 12), ("02", 20, 38), ("05", 20, 1)]:
        for kind, count in (("clean", n_clean), ("defective", n_bad)):
            for _ in range(count):
                out.append(Item(image_id=f"{n:04d}", fabric=fabric, kind=kind))
                n += 1
    return out


def test_grouped_every_image_is_tested_exactly_once(items):
    splits = grouped_kfold(items, k=5)
    tested = Counter(i for s in splits for i in s.test)
    assert set(tested) == {it.image_id for it in items}
    assert set(tested.values()) == {1}


def test_grouped_no_image_on_two_sides(items):
    for s in grouped_kfold(items, k=5):
        sides = [set(s.train), set(s.val), set(s.test)]
        assert not (sides[0] & sides[1] or sides[0] & sides[2] or sides[1] & sides[2])
        assert len(s.train) + len(s.val) + len(s.test) == len(items)


def test_grouped_is_stratified_and_balanced(items):
    kind = {it.image_id: (it.fabric, it.kind) for it in items}
    for s in grouped_kfold(items, k=5):
        counts = Counter(kind[i] for i in s.test)
        assert counts[("00", "clean")] == 4  # 20 images over 5 folds
        assert counts[("02", "defective")] in (7, 8)
        assert len(s.val) > 0 and any(kind[i][1] == "defective" for i in s.val)


def test_grouped_is_reproducible_and_seed_dependent(items):
    a, b = grouped_kfold(items, seed=1), grouped_kfold(items, seed=1)
    c = grouped_kfold(items, seed=2)
    assert [s.test for s in a] == [s.test for s in b]
    assert [s.test for s in a] != [s.test for s in c]


def test_lofo_holds_out_whole_fabric_and_nothing_else(items):
    fabric = {it.image_id: it.fabric for it in items}
    splits = leave_one_fabric_out(items)
    assert [s.fold for s in splits] == ["00", "01", "02", "05"]
    for s in splits:
        assert {fabric[i] for i in s.test} == {s.fold}
        assert s.fold not in {fabric[i] for i in s.train}
        assert s.fold not in {fabric[i] for i in s.val}
        assert set(s.test) == {i for i, f in fabric.items() if f == s.fold}
        s.assert_disjoint()


def test_validation_images_never_come_from_the_test_side(items):
    for s in grouped_kfold(items) + leave_one_fabric_out(items):
        assert not set(s.val) & set(s.test)


def test_leaky_split_leaks_and_grouped_split_does_not(items):
    starts = patch_starts(4096)
    patch_ids = [it.image_id for it in items for _ in starts]
    image_of_patch = np.array(patch_ids)

    for ps in leaky_patch_kfold(patch_ids, k=5):
        assert len(images_shared(patch_ids, ps.train, ps.test)) > 0  # measurable leakage
        assert not (set(ps.train) & set(ps.test))  # patch indices themselves are disjoint

    for s in grouped_kfold(items, k=5):
        train_idx = np.flatnonzero(np.isin(image_of_patch, list(s.train)))
        test_idx = np.flatnonzero(np.isin(image_of_patch, list(s.test)))
        assert images_shared(patch_ids, train_idx, test_idx) == set()


def test_leaky_split_covers_every_patch_once():
    ids = [f"{i // 31:04d}" for i in range(310)]
    tested = np.concatenate([ps.test for ps in leaky_patch_kfold(ids, k=5)])
    assert sorted(tested.tolist()) == list(range(310))


def test_threshold_picks_the_separating_value():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert best_f1_threshold(y, s) == 0.8


def test_threshold_needs_positives():
    with pytest.raises(ValueError):
        best_f1_threshold(np.zeros(4), np.arange(4.0))


def test_threshold_function_can_only_see_validation_arguments():
    params = list(inspect.signature(best_f1_threshold).parameters)
    assert params == ["y_val", "score_val"]
