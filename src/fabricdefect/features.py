"""Classical texture features, computed per patch.

Each strip is first normalised to a robust z-score (median and IQR-based scale), so that
features describe texture rather than the brightness of one fabric. Raw intensity
statistics are kept separately for the trivial baselines only.

Feature groups (what each captures):
  moments  spread and tails of the intensity distribution (a dark or bright thread
           shifts skew and kurtosis)
  glcm     grey-level co-occurrence statistics at pixel offsets 1, 2 and 4: contrast,
           dissimilarity, homogeneity, energy, correlation (regularity of the weave)
  lbp      local binary pattern histograms at two scales (micro-structure of yarns)
  gabor    responses of oriented band-pass filters at 4 frequencies x 4 orientations
           (is the periodic weave pattern locally intact?)
  fft      share of spectral energy in radial bands, total energy, and the ratio of
           horizontal to vertical frequency energy (weave period and direction)
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.signal import fftconvolve
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor_kernel

from .patches import PATCH_SIZE

GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 2, 4)
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
LBP_SETTINGS = ((8, 1), (16, 2))  # (points, radius)
GABOR_FREQS = (0.05, 0.1, 0.2, 0.3)
GABOR_THETAS = tuple(np.arange(4) * np.pi / 4)
FFT_BAND_EDGES = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.8)


def feature_names() -> list[str]:
    names = [f"mom_{n}" for n in ("std", "skew", "kurt", "p1", "p99")]
    names += [f"glcm_{p}_d{d}" for d in GLCM_DISTANCES for p in GLCM_PROPS]
    for pts, rad in LBP_SETTINGS:
        names += [f"lbp{pts}_{rad}_b{i}" for i in range(pts + 2)]
    names += [
        f"gabor_{stat}_f{f}_t{k}"
        for f in GABOR_FREQS
        for k in range(len(GABOR_THETAS))
        for stat in ("mean", "std")
    ]
    names += [f"fft_band{i}" for i in range(len(FFT_BAND_EDGES) - 1)]
    names += ["fft_logtotal", "fft_orient"]
    return names


def robust_normalise(img: np.ndarray) -> np.ndarray:
    """(x - median) / robust scale. The scale is the IQR / 1.349 (a std estimate that a
    small defect cannot move much)."""
    x = img.astype(np.float32)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    scale = (q3 - q1) / 1.349
    if scale < 1e-3:
        scale = max(float(x.std()), 1e-3)
    return (x - med) / scale


_KERNELS: list[tuple[np.ndarray, np.ndarray]] | None = None


def _gabor_kernels() -> list[tuple[np.ndarray, np.ndarray]]:
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = []
        for f in GABOR_FREQS:
            for th in GABOR_THETAS:
                k = gabor_kernel(f, theta=th)
                _KERNELS.append((k.real.astype(np.float32), k.imag.astype(np.float32)))
    return _KERNELS


def _radial_masks(size: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    r = np.sqrt(fx**2 + fy**2)
    bands = [
        (r >= lo) & (r < hi) for lo, hi in zip(FFT_BAND_EDGES[:-1], FFT_BAND_EDGES[1:], strict=True)
    ]
    ang = np.arctan2(np.abs(fy), fx)  # 0 = horizontal frequency axis
    horiz = (ang < np.pi / 8) & (r > 0)
    vert = (ang > 3 * np.pi / 8) & (r > 0)
    return bands, horiz, vert


def strip_features(
    img: np.ndarray, starts: list[int], size: int = PATCH_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features for every patch of one strip.

    Returns (features (n, d), raw_mean (n,), raw_std (n,)).
    """
    z = robust_normalise(img)
    n = len(starts)

    # strip-level maps, then windowed statistics per patch
    z_int = np.clip(np.round(z * 30 + 128), 0, 255).astype(np.uint8)  # LBP wants integers
    lbp_codes = [
        local_binary_pattern(z_int, p, r, "uniform").astype(np.int16) for p, r in LBP_SETTINGS
    ]
    gabor_mag = [
        np.hypot(fftconvolve(z, kr, mode="same"), fftconvolve(z, ki, mode="same"))
        for kr, ki in _gabor_kernels()
    ]

    bands, horiz, vert = _radial_masks(size)
    window = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    q = np.clip(((z + 3.0) / 6.0 * GLCM_LEVELS).astype(np.int64), 0, GLCM_LEVELS - 1).astype(
        np.uint8
    )

    feats = np.zeros((n, len(feature_names())), dtype=np.float32)
    raw_mean = np.zeros(n, dtype=np.float32)
    raw_std = np.zeros(n, dtype=np.float32)
    for i, x in enumerate(starts):
        sl = slice(x, x + size)
        pz = z[:, sl]
        pz64 = pz.astype(np.float64)
        raw = img[:, sl].astype(np.float32)
        raw_mean[i], raw_std[i] = raw.mean(), raw.std()
        with warnings.catch_warnings():
            # Blank margins give constant patches, where skew and kurtosis are numerically
            # meaningless; the values are kept as they were and zeroed by nan_to_num below.
            warnings.simplefilter("ignore", RuntimeWarning)
            skew = stats.skew(pz64, axis=None)
            kurt = stats.kurtosis(pz64, axis=None)
        row = [pz64.std(), skew, kurt, *np.percentile(pz, [1, 99])]
        glcm = graycomatrix(
            q[:, sl],
            distances=GLCM_DISTANCES,
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=GLCM_LEVELS,
            symmetric=True,
            normed=True,
        )
        for d in range(len(GLCM_DISTANCES)):
            for p in GLCM_PROPS:
                row.append(np.nan_to_num(graycoprops(glcm, p)[d].mean()))
        for (pts, _), codes in zip(LBP_SETTINGS, lbp_codes, strict=True):
            hist = np.bincount(codes[:, sl].ravel(), minlength=pts + 2)[: pts + 2]
            row.extend(hist / hist.sum())
        for mag in gabor_mag:
            w = mag[:, sl]
            row.extend([w.mean(), w.std()])
        power = np.abs(np.fft.rfft2(pz * window)) ** 2
        total = power.sum() + 1e-12
        row.extend(power[b].sum() / total for b in bands)
        row.append(np.log(total))
        row.append(np.log((power[horiz].sum() + 1e-12) / (power[vert].sum() + 1e-12)))
        feats[i] = np.nan_to_num(np.asarray(row, dtype=np.float32))
    return feats, raw_mean, raw_std
