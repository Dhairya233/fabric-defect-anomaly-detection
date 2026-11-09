"""Run every method on every split and write the result tables to results/.

Needs data/patch_data.npz (run scripts/extract_features.py first).
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn

from fabricdefect.cli import require
from fabricdefect.experiments import SPLITS, Run, run_protocol
from fabricdefect.patchdata import CACHE_PATH, PatchData
from fabricdefect.reporting import build_tables


def save_runs(runs, pred_path: Path, info_path: Path) -> None:
    np.savez_compressed(
        pred_path,
        **{
            f"{s}|{m}|{what}": getattr(r, what)
            for (s, m), r in runs.items()
            if r is not None
            for what in ("scores", "thr")
        },
    )
    cells = [
        {"split": s, "method": m, "applicable": r is not None, "info": r.info if r else {}}
        for (s, m), r in runs.items()
    ]
    info_path.write_text(json.dumps(cells))


def load_runs(pred_path: Path, info_path: Path) -> dict:
    z = np.load(pred_path)
    runs = {}
    for c in json.loads(info_path.read_text()):
        key = (c["split"], c["method"])
        runs[key] = (
            Run(z[f"{key[0]}|{key[1]}|scores"], z[f"{key[0]}|{key[1]}|thr"], c["info"])
            if c["applicable"]
            else None
        )
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch-data", type=Path, default=CACHE_PATH)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples of images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="skip fitting and reuse data/cache/predictions.npz from an earlier run",
    )
    ap.add_argument(
        "--mask-margins",
        action="store_true",
        help="sensitivity run: leave patches that touch a blank margin out of fitting and "
        "evaluation (needs scripts/find_margins.py); use with --out and --tag",
    )
    ap.add_argument("--tag", default="", help="suffix for the cached prediction files")
    args = ap.parse_args()

    require(args.patch_data, "Run python scripts/extract_features.py first.")
    pdat = PatchData.load(args.patch_data)
    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    valid = None
    if args.mask_margins:
        require(cache / "margin_patch.npy", "Run python scripts/find_margins.py first.")
        valid = ~np.load(cache / "margin_patch.npy")
    pred_path = cache / f"predictions{args.tag}.npz"
    info_path = cache / f"run_cells{args.tag}.json"
    if args.reuse_predictions:
        runs = load_runs(pred_path, info_path)
    else:
        t0 = time.time()
        runs = run_protocol(pdat, progress=print, valid=valid)
        print(f"fitting and scoring done in {time.time() - t0:.0f} s")
        save_runs(runs, pred_path, info_path)
    print("bootstrapping...")

    tables = build_tables(pdat, runs, n_boot=args.boot, seed=args.seed, valid=valid)
    args.out.mkdir(exist_ok=True)
    for name, df in tables.items():
        df.to_csv(args.out / f"{name}.csv", index=False, float_format="%.4f")

    info = {
        "seed": args.seed,
        "bootstrap_resamples": args.boot,
        "bootstrap_unit": "whole images",
        "margin_patches_excluded": bool(args.mask_margins),
        "splits": list(SPLITS),
        "methods": sorted({m for (_, m) in runs}),
        "not_applicable": [f"{s}: {m}" for (s, m), r in runs.items() if r is None],
        "configurations_evaluated_on_test": sum(r is not None for r in runs.values()),
        "svm_C_selected_per_fold": {
            s: r.info["C"] for (s, m), r in runs.items() if r is not None and r.info["C"]
        },
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
    }
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"wrote tables to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
