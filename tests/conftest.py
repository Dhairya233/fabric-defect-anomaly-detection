import numpy as np
import pytest

from fabricdefect.patchdata import PatchData


def make_pdat(n_fabrics=4, n_clean=8, n_defective=5, dim=6, seed=0) -> PatchData:
    """Small synthetic dataset: 512-px strips with 3 patches each. Defective strips have
    a defect in the right two patches; their features are shifted along feature 0."""
    rng = np.random.default_rng(seed)
    ids, fabrics, kinds, codes = [], [], [], []
    px, lab, feats, rmean, rstd, cols = [], [], [], [], [], []
    n = 0
    for f in range(n_fabrics):
        for kind, count in (("clean", n_clean), ("defective", n_defective)):
            for _ in range(count):
                ids.append(f"{n:04d}_{19 if kind == 'defective' else 0:03d}_{f:02d}")
                fabrics.append(f"{f:02d}")
                kinds.append(kind)
                codes.append(19 if kind == "defective" else 0)
                bad = kind == "defective"
                px.extend([0, 30, 30] if bad else [0, 0, 0])
                lab.extend([0, 1, 1] if bad else [0, 0, 0])
                x = rng.normal(size=(3, dim)) + f  # fabric-specific offset
                if bad:
                    x[1:, 0] += 4.0
                feats.append(x)
                rmean.extend(rng.normal(100 + 10 * f, 3, 3))
                rstd.extend(rng.normal(10, 1, 3) + (3 if bad else 0))
                c = np.zeros(512, dtype=np.int16)
                if bad:
                    c[200:230] = 5
                cols.append(c)
                n += 1
    return PatchData(
        image_ids=np.array(ids),
        fabrics=np.array(fabrics),
        kinds=np.array(kinds),
        codes=np.array(codes),
        widths=np.full(n, 512),
        patch_offsets=np.arange(n + 1) * 3,
        col_offsets=np.arange(n + 1) * 512,
        col_defect=np.concatenate(cols),
        patch_x=np.tile([0, 128, 256], n),
        patch_px=np.array(px),
        patch_label=np.array(lab, dtype=np.int8),
        features=np.concatenate(feats).astype(np.float32),
        raw_mean=np.array(rmean, dtype=np.float32),
        raw_std=np.array(rstd, dtype=np.float32),
        extract_seconds=np.full(n, 0.01),
        feature_names=np.array([f"f{i}" for i in range(dim)]),
    )


@pytest.fixture
def pdat():
    return make_pdat()
