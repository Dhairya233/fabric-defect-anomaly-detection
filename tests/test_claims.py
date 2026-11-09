"""The headline numbers quoted in the README and on the site must match the tables."""

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _rows(rel):
    with open(ROOT / rel) as f:
        return list(csv.DictReader(f))


def _gap(method, comparison, metric):
    for r in _rows("results/margin_masked/leakage_gap.csv"):
        if (r["method"], r["comparison"], r["metric"]) == (method, comparison, metric):
            return float(r["difference"]), float(r["ci_low"]), float(r["ci_high"])
    raise KeyError


def _metric(folder, method, split, metric):
    for r in _rows(f"{folder}/metrics_overall.csv"):
        if (r["method"], r["split"], r["metric"]) == (method, split, metric):
            return float(r["value"])
    raise KeyError


def test_leakage_headline_for_the_random_forest():
    d, lo, hi = _gap("M1 random forest", "leaky minus grouped", "patch_ap")
    assert (round(d, 2), round(lo, 2), round(hi, 2)) == (0.44, 0.27, 0.53)
    grouped = _metric("results/margin_masked", "M1 random forest", "grouped", "patch_ap")
    leaky = _metric("results/margin_masked", "M1 random forest", "leaky", "patch_ap")
    assert (round(grouped, 2), round(leaky, 2)) == (0.23, 0.68)
    readme = (ROOT / "README.md").read_text()
    assert "0.23 [0.16, 0.38]" in readme and "0.68 [0.50, 0.80]" in readme


def test_margin_effect_on_the_best_pooled_score():
    method = "M2 Mahalanobis (per fabric)"
    assert round(_metric("results", method, "grouped", "image_auroc"), 3) == 0.929
    assert round(_metric("results/margin_masked", method, "grouped", "image_auroc"), 3) == 0.873


def test_unseen_fabric_headline_is_the_best_of_eight_methods():
    rows = [
        r for r in _rows("results/margin_masked/artefact_free_subset.csv") if r["split"] == "lofo"
    ]
    best = max(rows, key=lambda r: float(r["mean_per_fabric_image_auroc"]))
    assert best["method"] == "M1 random forest"
    got = tuple(
        round(float(best[k]), 2) for k in ("mean_per_fabric_image_auroc", "ci_low", "ci_high")
    )
    assert got == (0.59, 0.44, 0.72)


@pytest.mark.parametrize("path", ["README.md", "site/index.html", "docs/results.md"])
def test_quoted_numbers_appear_in_the_text(path):
    text = (ROOT / path).read_text()
    assert "0.59" in text and "0.72" in text
    if path == "README.md":
        assert "+0.44" in text and "0.27 to 0.53" in text


def test_every_table_value_in_the_readme_comes_from_the_csvs():
    text = (ROOT / "README.md").read_text()
    for r in _rows("results/margin_masked/leakage_gap.csv"):
        if r["metric"] in ("patch_ap", "image_macro_fabric_auroc") and r["comparison"] in (
            "leaky minus grouped",
            "leaky minus lofo",
        ):
            d, lo, hi = float(r["difference"]), float(r["ci_low"]), float(r["ci_high"])
            if r["method"] in ("M1 random forest", "M1 SVM (RBF)", "M2 Mahalanobis (global)"):
                assert f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]" in text, r
    for r in _rows("results/margin_masked/metrics_overall.csv"):
        if r["metric"] == "image_macro_fabric_auroc" and r["method"] != "M0 always clean":
            v, lo, hi = float(r["value"]), float(r["ci_low"]), float(r["ci_high"])
            assert f"{v:.2f} [{lo:.2f}, {hi:.2f}]" in text, r
    for r in _rows("results/margin_masked/artefact_free_subset.csv"):
        if r["method"] == "M0 always clean":
            continue
        v, lo, hi = (float(r[k]) for k in ("mean_per_fabric_image_auroc", "ci_low", "ci_high"))
        assert f"{v:.2f} [{lo:.2f}, {hi:.2f}]" in text, r


def test_readme_counts_match_the_tables():
    text = (ROOT / "README.md").read_text()
    with open(ROOT / "results/margin_summary.csv") as fh:
        m = {r["quantity"]: r["value"] for r in csv.DictReader(fh)}
    assert (
        f"{m['strips_with_blank_columns']} of {m['strips']} strips have blank white margins" in text
    )
    first = _rows("results/margin_masked/metrics_overall.csv")[0]
    defective = int(float(first["n_defective_patches"]))
    labelled = int(float(first["n_labelled_patches"]))
    assert f"{defective} of {labelled:,} labelled" in text
