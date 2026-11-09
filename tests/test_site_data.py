"""The site reads precomputed JSON; these tests keep it in step with the result tables."""

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEY_TO_NAME = {
    "contrast": "M0 raw contrast",
    "brightness": "M0 brightness deviation",
    "rf": "M1 random forest",
    "svm": "M1 SVM (RBF)",
    "maha_global": "M2 Mahalanobis (global)",
    "maha_fabric": "M2 Mahalanobis (per fabric)",
    "iforest": "M2 isolation forest (global)",
    "ocsvm": "M2 one-class SVM (global)",
}


@pytest.fixture(scope="module")
def strips():
    return json.loads((ROOT / "site/data/strips.json").read_text())


@pytest.fixture(scope="module")
def results():
    return json.loads((ROOT / "site/data/results.json").read_text())


def _csv_value(folder, method, split, metric):
    with open(ROOT / folder / "metrics_overall.csv") as f:
        for r in csv.DictReader(f):
            if (r["method"], r["split"], r["metric"]) == (method, split, metric):
                return float(r["value"])
    raise KeyError((method, split, metric))


def test_every_strip_is_present_once(strips):
    ids = [s["id"] for s in strips["strips"]]
    assert len(ids) == len(set(ids)) == 245
    assert sum(s["defective"] for s in strips["strips"]) == 105
    assert set(strips["methods"]) == set(KEY_TO_NAME)


def test_strip_scores_have_one_value_per_bin(strips):
    for s in strips["strips"]:
        for m in strips["methods"]:
            assert len(s["scores"][m]["v"]) == strips["bins"]
            assert all(v is None or -100 <= v <= 100 for v in s["scores"][m]["v"])


def test_flag_counts_match_recall_and_false_alarms_in_the_tables(strips):
    defective = [s for s in strips["strips"] if s["defective"]]
    for key, name in KEY_TO_NAME.items():
        flagged = sum(s["scores"][key]["flag"] for s in defective)
        recall = _csv_value("results/margin_masked", name, "grouped", "image_recall")
        assert flagged / 105 == pytest.approx(recall, abs=1e-3), name


def test_table_values_match_the_csv(results):
    for key, name in KEY_TO_NAME.items():
        got = results["runs"]["masked"]["table"][key]["grouped"]["image_auroc"][0]
        want = _csv_value("results/margin_masked", name, "grouped", "image_auroc")
        assert got == pytest.approx(want, abs=1e-3), name


def test_unavailable_cell_is_marked_not_applicable(results):
    assert "lofo:maha_fabric" in results["runs"]["masked"]["not_applicable"]
