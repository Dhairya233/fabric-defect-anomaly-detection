import numpy as np
import pytest
from PIL import Image

from fabricdefect.download import sha256_of, verify_archives
from fabricdefect.manifest import build_manifest, mask_owner, parse_image_id


def _png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def raw(tmp_path):
    base = np.full((8, 32), 100, dtype=np.uint8)
    mask = np.zeros((8, 32), dtype=np.uint8)
    mask[2:4, 5:9] = 255
    _png(tmp_path / "NODefect_images/NODefect_images/f/0001_000_00.png", base)
    _png(tmp_path / "NODefect_images/NODefect_images/f/0018_00_01.png", base)  # excluded
    _png(tmp_path / "Defect_images/Defect_images/0002_019_00.png", base)
    _png(tmp_path / "Defect_images/Defect_images/0003_030_00.png", base)
    _png(tmp_path / "Defect_images/Defect_images/0100_025_08.png", base)  # excluded
    _png(tmp_path / "Mask_images/Mask_images/0002_019_00_mask.png", mask)
    _png(tmp_path / "Mask_images/Mask_images/0003_030_00_mask1.png", mask * 0)
    _png(tmp_path / "Mask_images/Mask_images/0003_030_00_mask2.png", mask)
    return tmp_path


def test_parse_image_id():
    assert parse_image_id("0044_019_04") == ("0044", 19, "04")
    assert parse_image_id("0018_00_01") == ("0018", 0, "01")
    assert parse_image_id("garbage") is None


def test_mask_owner_handles_numbered_masks():
    assert mask_owner("0044_019_04_mask") == "0044_019_04"
    assert mask_owner("0044_019_04_mask2") == "0044_019_04"
    assert mask_owner("0044_019_04") is None


def test_manifest_applies_exclusions_and_pairs_masks(raw):
    recs = {r.image_id: r for r in build_manifest(raw)}
    assert set(recs) == {"0001_000_00", "0002_019_00", "0003_030_00"}
    assert recs["0001_000_00"].kind == "clean"
    assert recs["0001_000_00"].mask_paths == ""
    assert recs["0002_019_00"].defect_name == "fuzzyball"
    assert len(recs["0003_030_00"].mask_paths.split(";")) == 2
    assert not recs["0003_030_00"].mask_empty  # union of an empty and a non-empty mask


def test_empty_mask_is_flagged(raw):
    _png(raw / "Mask_images/Mask_images/0002_019_00_mask.png", np.zeros((8, 32), np.uint8))
    recs = {r.image_id: r for r in build_manifest(raw)}
    assert recs["0002_019_00"].mask_empty


def test_defective_image_without_mask_is_an_error(raw):
    (raw / "Mask_images/Mask_images/0002_019_00_mask.png").unlink()
    with pytest.raises(ValueError, match="without a mask"):
        build_manifest(raw)


def test_verify_archives(tmp_path):
    f = tmp_path / "a.7z"
    f.write_bytes(b"abc")
    good = {"a.7z": sha256_of(f)}
    assert verify_archives(tmp_path, good) == []
    assert verify_archives(tmp_path, {"a.7z": "0" * 64})[0].startswith("checksum mismatch")
    assert verify_archives(tmp_path, {"b.7z": "0" * 64})[0].startswith("missing")
