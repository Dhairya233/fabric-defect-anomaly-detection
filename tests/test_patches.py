import numpy as np
import pytest
from PIL import Image

from fabricdefect.data import load_image, load_mask
from fabricdefect.patches import (
    AMBIGUOUS,
    CLEAN,
    DEFECTIVE,
    assemble_score_map,
    defect_pixels_per_patch,
    extract_patches,
    label_patches,
    patch_starts,
)


def test_full_width_strip_has_31_patches():
    starts = patch_starts(4096)
    assert len(starts) == 31
    assert starts[0] == 0 and starts[-1] == 3840


def test_odd_width_gets_an_edge_aligned_final_patch():
    starts = patch_starts(3796)
    assert starts[-1] == 3796 - 256
    assert starts[-1] - starts[-2] <= 128


@pytest.mark.parametrize("width", [256, 257, 300, 3796, 4096])
def test_patches_cover_every_column(width):
    covered = np.zeros(width, dtype=bool)
    for x in patch_starts(width):
        covered[x : x + 256] = True
    assert covered.all()


def test_narrow_strip_is_rejected():
    with pytest.raises(ValueError):
        patch_starts(200)


def test_extract_patches_matches_slices():
    img = np.arange(256 * 600, dtype=np.int64).reshape(256, 600) % 251
    starts = patch_starts(600)
    patches = extract_patches(img.astype(np.uint8), starts)
    assert patches.shape == (len(starts), 256, 256)
    for p, x in zip(patches, starts, strict=True):
        assert np.array_equal(p, img[:, x : x + 256])


def test_extract_patches_requires_square_height():
    with pytest.raises(ValueError):
        extract_patches(np.zeros((100, 600), np.uint8), [0])


def test_defect_pixels_and_labels():
    mask = np.zeros((256, 640), dtype=bool)
    mask[10:12, 300:303] = True  # 6 px, inside patches starting at 128 and 256
    starts = patch_starts(640)
    px = defect_pixels_per_patch(mask, starts)
    assert px.tolist() == [0, 6, 6, 0]
    assert label_patches(px, 5).tolist() == [CLEAN, DEFECTIVE, DEFECTIVE, CLEAN]
    assert label_patches(px, 10).tolist() == [CLEAN, AMBIGUOUS, AMBIGUOUS, CLEAN]


def test_assemble_constant_scores_gives_constant_map():
    starts = patch_starts(900)
    m = assemble_score_map(np.full(len(starts), 0.7), starts, 900)
    assert m.shape == (256, 900)
    assert np.allclose(m, 0.7)


def test_assemble_averages_overlaps_and_keeps_unique_regions():
    starts = [0, 128]
    m = assemble_score_map(np.array([0.0, 1.0]), starts, 384)
    assert np.allclose(m[:, :128], 0.0)  # only the first patch
    assert np.allclose(m[:, 128:256], 0.5)  # overlap
    assert np.allclose(m[:, 256:], 1.0)  # only the second patch


def test_assemble_rejects_incomplete_coverage():
    with pytest.raises(ValueError):
        assemble_score_map(np.array([1.0]), [0], 600)


def test_load_image_drops_alpha(tmp_path):
    p = tmp_path / "a.png"
    Image.fromarray(np.full((4, 8, 2), [90, 255], dtype=np.uint8), mode="LA").save(p)
    img = load_image(p)
    assert img.shape == (4, 8) and img.dtype == np.uint8 and (img == 90).all()


def test_load_mask_thresholds_and_takes_union(tmp_path):
    a = np.zeros((4, 8), np.uint8)
    a[0, 0] = 200  # defect
    a[0, 1] = 100  # anti-aliasing residue, below threshold
    b = np.zeros((4, 8), np.uint8)
    b[3, 7] = 255
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(b).save(tmp_path / "b.png")
    m = load_mask([tmp_path / "a.png", tmp_path / "b.png"], (4, 8))
    assert m.sum() == 2 and m[0, 0] and m[3, 7] and not m[0, 1]
    assert not load_mask([], (4, 8)).any()


def test_load_mask_rejects_wrong_shape(tmp_path):
    Image.fromarray(np.zeros((4, 8), np.uint8)).save(tmp_path / "a.png")
    with pytest.raises(ValueError):
        load_mask([tmp_path / "a.png"], (5, 8))
