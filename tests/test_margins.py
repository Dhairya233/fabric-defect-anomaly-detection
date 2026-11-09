import numpy as np
import pytest

from fabricdefect.evaluation import prepare
from fabricdefect.experiments import build_folds, run_method
from fabricdefect.margins import margin_columns, margin_patch_flags
from fabricdefect.models import RandomForest, Scorer
from fabricdefect.patches import patch_starts


def _strip(blank_cols: int, width: int = 1024) -> np.ndarray:
    img = np.full((256, width), 90, dtype=np.uint8)
    img[:, :blank_cols] = 255
    return img


def test_margin_columns_cover_blank_plus_safety_band():
    cols = margin_columns(_strip(100), dilation=20)
    assert cols[:120].all() and not cols[121:].any()


def test_strip_without_margin_is_untouched():
    assert not margin_columns(_strip(0)).any()
    assert not margin_patch_flags(_strip(0), patch_starts(1024)).any()


def test_only_patches_touching_the_margin_are_flagged():
    starts = patch_starts(1024)  # 0, 128, ..., 768
    flags = margin_patch_flags(_strip(100), starts, dilation=20)
    # margin columns end at 120: only the window 0-256 touches them; 128-384 starts after
    assert flags.tolist() == [True] + [False] * (len(starts) - 1)


class _Recorder(Scorer):
    """Remembers how many patches it was fitted on and scores everything as 'defective'."""

    name = "recorder"
    seen: list[int] = []

    def fit(self, train, val):
        _Recorder.seen.append(len(train.y))

    def score(self, view):
        return np.ones(len(view.y))


def test_invalid_patches_are_not_trained_on_and_never_alarm(pdat):
    n = len(pdat.patch_x)
    valid = np.ones(n, bool)
    valid[::5] = False  # every fifth patch is unusable
    folds = build_folds(pdat)["grouped"]
    _Recorder.seen = []
    run = run_method(pdat, folds, _Recorder, valid=valid)
    assert _Recorder.seen == [int(valid[f.train].sum()) for f in folds]
    alarms = run.scores >= run.thr
    assert not alarms[~valid].any()
    assert alarms[valid].all()


def test_random_forest_still_works_with_a_valid_mask(pdat):
    valid = np.ones(len(pdat.patch_x), bool)
    valid[::7] = False
    run = run_method(pdat, build_folds(pdat)["grouped"], RandomForest, valid=valid)
    assert not np.isnan(run.scores).any()


def test_patch_metrics_skip_invalid_patches(pdat):
    n = len(pdat.patch_x)
    scores, thr = pdat.patch_label.clip(min=0).astype(float), np.full(n, 0.5)
    valid = np.ones(n, bool)
    valid[::4] = False
    assert len(prepare(pdat, scores, thr, valid).patch_y) < len(prepare(pdat, scores, thr).patch_y)


def test_saturated_fraction_counts_blank_columns():
    from fabricdefect.margins import saturated_fraction

    assert saturated_fraction(_strip(100)) == pytest.approx(100 / 1024)
    assert saturated_fraction(_strip(0)) == 0.0
