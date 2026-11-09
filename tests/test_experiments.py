import numpy as np
import pytest

from fabricdefect.experiments import build_folds, run_method, run_protocol
from fabricdefect.models import (
    SVM,
    Mahalanobis,
    MajorityClass,
    NotApplicable,
    RandomForest,
    method_factories,
)
from fabricdefect.thresholds import best_f1_threshold


def test_folds_never_mix_test_with_train_or_val(pdat):
    for split, folds in build_folds(pdat, k=5).items():
        tested = np.concatenate([f.test for f in folds])
        assert sorted(tested.tolist()) == list(range(len(pdat.patch_x)))  # each patch once
        for f in folds:
            assert not set(f.test) & set(f.train)
            assert not set(f.test) & set(f.val)
            assert not set(f.train) & set(f.val)
            if split != "leaky":  # whole images only
                img = pdat.patch_image
                assert not set(img[f.test]) & set(img[f.train])
                assert not set(img[f.test]) & set(img[f.val])


def test_threshold_is_chosen_on_validation_patches_only(pdat):
    folds = build_folds(pdat, k=5)["grouped"]
    seen = []

    def spy(y_val, score_val):
        seen.append(len(y_val))
        return best_f1_threshold(y_val, score_val)

    run = run_method(pdat, folds, RandomForest, threshold_fn=spy)
    expected = [int((pdat.patch_label[f.val] >= 0).sum()) for f in folds]
    assert seen == expected  # one call per fold, sized exactly like that fold's validation set
    assert not np.isnan(run.thr).any()


def test_supervised_and_anomaly_methods_beat_chance_on_separable_data(pdat):
    folds = build_folds(pdat, k=5)["grouped"]
    from sklearn.metrics import roc_auc_score

    lab = pdat.patch_label >= 0
    for factory in (RandomForest, SVM, lambda: Mahalanobis(per_fabric=False)):
        run = run_method(pdat, folds, factory)
        assert roc_auc_score(pdat.patch_label[lab] == 1, run.scores[lab]) > 0.8


def test_majority_class_never_alarms(pdat):
    run = run_method(pdat, build_folds(pdat)["grouped"], MajorityClass)
    assert not (run.scores >= run.thr).any()


def test_per_fabric_model_is_not_applicable_on_unseen_fabrics(pdat):
    folds = build_folds(pdat)
    with pytest.raises(NotApplicable):
        run_method(pdat, folds["lofo"], lambda: Mahalanobis(per_fabric=True))
    run_method(pdat, folds["grouped"], lambda: Mahalanobis(per_fabric=True))


def test_full_protocol_runs_and_marks_inapplicable_cells(pdat):
    runs = run_protocol(pdat)
    assert len(runs) == 3 * len(method_factories())
    assert runs[("lofo", "M2 Mahalanobis (per fabric)")] is None
    assert runs[("grouped", "M2 Mahalanobis (per fabric)")] is not None
