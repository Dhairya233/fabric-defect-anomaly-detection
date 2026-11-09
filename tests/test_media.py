"""The demo media must be built from the result tables, never from typed-in numbers."""

import csv
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "media"))

import facts  # noqa: E402


def _metric(split, metric, method="M1 random forest"):
    with open(ROOT / "results/margin_masked/metrics_overall.csv") as f:
        for r in csv.DictReader(f):
            if (r["method"], r["split"], r["metric"]) == (method, split, metric):
                return float(r["value"])
    raise KeyError


def test_every_placeholder_in_the_script_is_a_known_fact():
    script = json.loads((ROOT / "media/script.json").read_text())
    known = set(facts.load())
    for video in ("main", "short"):
        for scene in script[video]["scenes"]:
            for line in scene["lines"]:
                for name in re.findall(r"\{(\w+)\}", line["text"]):
                    assert name in known, f"{video}/{line['id']}: unknown placeholder {name}"


def test_spoken_numbers_match_the_result_tables():
    f = facts.load()
    assert f["rf_ap_grouped"] == f"{_metric('grouped', 'patch_ap'):.2f}"
    assert f["rf_ap_leaky"] == f"{_metric('leaky', 'patch_ap'):.2f}"
    ratio = _metric("leaky", "patch_ap") / _metric("grouped", "patch_ap")
    assert f["ratio"] == f"{ratio:.1f}"


def test_margin_count_comes_from_the_regenerated_summary():
    with open(ROOT / "results/margin_summary.csv") as fh:
        rows = {r["quantity"]: r["value"] for r in csv.DictReader(fh)}
    assert facts.load()["margin_strips"] == rows["strips_with_blank_columns"]


def test_caption_chunks_are_balanced():
    pytest.importorskip("playwright")
    pytest.importorskip("soundfile")
    import build

    text = "The same random forest scored 0.23 or 0.68, depending only on how the data was split."
    pieces = build.chunks(text)
    assert " ".join(pieces) == text
    assert all(len(p) <= 72 for p in pieces)
    assert min(len(p.split()) for p in pieces) >= 4  # no one-word orphans
    assert build.ts(3661.5) == "01:01:01,500"
