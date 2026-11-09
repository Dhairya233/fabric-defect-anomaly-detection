"""Threshold selection. Takes validation data only, by construction: there is no way to
pass test scores into these functions without writing it out explicitly."""

from __future__ import annotations

import numpy as np


def best_f1_threshold(y_val: np.ndarray, score_val: np.ndarray) -> float:
    """Score threshold (predict defective when score >= threshold) maximising F1 on the
    validation data. Ties go to the higher threshold, which gives fewer alarms."""
    y = np.asarray(y_val).astype(bool)
    s = np.asarray(score_val, dtype=float)
    if y.shape != s.shape or y.size == 0:
        raise ValueError("y_val and score_val must be non-empty and the same shape")
    if not y.any():
        raise ValueError("validation data has no defective examples; cannot choose a threshold")
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    # Only cut between distinct score values.
    last_of_value = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    tp, fp, cand = tp[last_of_value], fp[last_of_value], s_sorted[last_of_value]
    f1 = 2 * tp / (2 * tp + fp + (y.sum() - tp))
    return float(cand[int(np.argmax(f1))])
