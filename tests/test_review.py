from types import SimpleNamespace

import numpy as np
import pytest

from fabricdefect.cli import require
from fabricdefect.evaluation import count_matched_image_scores


def _stub(counts):
    """Strips with the given numbers of patches, all patches scored."""
    offsets = np.r_[0, np.cumsum(counts)]
    return SimpleNamespace(
        n_images=len(counts),
        patches_of=lambda i: slice(int(offsets[i]), int(offsets[i + 1])),
    ), int(offsets[-1])


def test_constant_scores_are_unchanged_by_matching():
    pdat, n = _stub([6, 4, 9])
    scores = np.concatenate([np.full(6, 1.0), np.full(4, 2.0), np.full(9, 3.0)])
    out = count_matched_image_scores(pdat, scores, np.ones(n, bool), m=4)
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_matching_removes_the_advantage_of_having_more_patches():
    rng = np.random.default_rng(0)
    pdat, n = _stub([30, 10] * 40)
    scores = rng.normal(size=n)  # identical distribution for every strip
    valid = np.ones(n, bool)
    raw = np.array([scores[pdat.patches_of(i)].max() for i in range(pdat.n_images)])
    matched = count_matched_image_scores(pdat, scores, valid, m=10, n_draws=100)
    raw_gap = raw[0::2].mean() - raw[1::2].mean()  # 30-patch strips against 10-patch strips
    matched_gap = matched[0::2].mean() - matched[1::2].mean()
    assert raw_gap > 0.5
    assert abs(matched_gap) < 0.25


def test_matching_ignores_unscored_patches_and_checks_the_minimum():
    pdat, n = _stub([5, 5])
    scores = np.array([9.0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    valid = np.ones(n, bool)
    valid[0] = False  # the 9.0 patch is not scored
    out = count_matched_image_scores(pdat, scores, valid, m=4)
    assert np.allclose(out, [1.0, 1.0])
    with pytest.raises(ValueError):
        count_matched_image_scores(pdat, scores, valid, m=5)


def test_require_exits_with_a_plain_message(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        require(tmp_path / "missing.npz", "Run the earlier step first.")
    assert e.value.code == 1
    assert "Run the earlier step first." in capsys.readouterr().err
    assert require(tmp_path, "unused") == tmp_path
