"""Scoring methods. Every method turns a patch into one number, higher = more defective.

M0  trivial baselines (always "clean"; raw contrast; raw brightness deviation)
M1  supervised: Random Forest and RBF-SVM on texture features, trained on patch labels
M2  anomaly: models of defect-free fabric only (Mahalanobis, isolation forest, one-class
    SVM); a patch is scored by how unlike normal fabric it looks
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM

from .splits import SEED


class NotApplicable(Exception):
    """The method cannot be run for this split (for example a per-fabric model on a
    fabric it was never trained on)."""


@dataclass
class View:
    """The patches one model is allowed to see at one stage."""

    X: np.ndarray  # texture features
    y: np.ndarray  # patch labels: 1 defective, 0 clean, -1 ambiguous
    raw_mean: np.ndarray
    raw_std: np.ndarray
    fabric: np.ndarray
    clean_image: np.ndarray  # True where the patch comes from a defect-free strip


class Scorer:
    name = ""
    fixed_threshold: float | None = None  # set by methods that must not use a tuned one
    info: dict

    def __init__(self):
        self.info = {}

    def fit(self, train: View, val: View) -> None:
        raise NotImplementedError

    def score(self, view: View) -> np.ndarray:
        raise NotImplementedError


class MajorityClass(Scorer):
    """Predicts 'clean' for everything: the right answer for ~57% of strips."""

    name, fixed_threshold = "M0 always clean", 1.0

    def fit(self, train, val):
        pass

    def score(self, view):
        return np.zeros(len(view.y))


class RawContrast(Scorer):
    """Score = standard deviation of raw pixel intensity in the patch."""

    name = "M0 raw contrast"

    def fit(self, train, val):
        pass

    def score(self, view):
        return view.raw_std.astype(float)


class RawBrightnessDeviation(Scorer):
    """Score = distance of the patch's mean brightness from the average clean patch."""

    name = "M0 brightness deviation"

    def fit(self, train, val):
        m = train.raw_mean[train.clean_image]
        self.mu, self.sd = float(m.mean()), float(m.std()) + 1e-9

    def score(self, view):
        return np.abs(view.raw_mean - self.mu) / self.sd


def _labelled(v: View) -> np.ndarray:
    return v.y >= 0


class RandomForest(Scorer):
    name = "M1 random forest"

    def fit(self, train, val):
        m = _labelled(train)
        self.model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=1,
        ).fit(train.X[m], train.y[m])

    def score(self, view):
        return self.model.predict_proba(view.X)[:, 1]


class SVM(Scorer):
    """RBF SVM. C is the only hyper-parameter tuned, on validation patches (3 values)."""

    name = "M1 SVM (RBF)"
    C_GRID = (0.1, 1.0, 10.0)

    def fit(self, train, val):
        m, mv = _labelled(train), _labelled(val)
        self.scaler = StandardScaler().fit(train.X[m])
        Xt, yt = self.scaler.transform(train.X[m]), train.y[m]
        best_c, best_ap = 1.0, -1.0
        for c in self.C_GRID:
            clf = SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced").fit(Xt, yt)
            if val.y[mv].sum() == 0:
                break
            ap = average_precision_score(
                val.y[mv], clf.decision_function(self.scaler.transform(val.X[mv]))
            )
            if ap > best_ap:
                best_c, best_ap, self.model = c, ap, clf
        if best_ap < 0:
            self.model = SVC(C=best_c, kernel="rbf", gamma="scale", class_weight="balanced").fit(
                Xt, yt
            )
        self.info["C"] = best_c

    def score(self, view):
        return self.model.decision_function(self.scaler.transform(view.X))


class _AnomalyBase(Scorer):
    def _normal(self, train: View) -> np.ndarray:
        """Training patches: defect-free strips only."""
        return train.clean_image & (train.y == 0)


class Mahalanobis(_AnomalyBase):
    """Squared Mahalanobis distance to the mean of normal patches (shrinkage covariance).
    per_fabric=True fits one model per training fabric and needs to know the patch's fabric."""

    def __init__(self, per_fabric: bool):
        super().__init__()
        self.per_fabric = per_fabric
        self.name = "M2 Mahalanobis (per fabric)" if per_fabric else "M2 Mahalanobis (global)"

    def _fit_one(self, X):
        scaler = StandardScaler().fit(X)
        return scaler, LedoitWolf().fit(scaler.transform(X))

    def fit(self, train, val):
        normal = self._normal(train)
        if self.per_fabric:
            self.models = {
                f: self._fit_one(train.X[normal & (train.fabric == f)])
                for f in np.unique(train.fabric[normal])
            }
        else:
            self.models = {None: self._fit_one(train.X[normal])}

    def score(self, view):
        if not self.per_fabric:
            sc, lw = self.models[None]
            return lw.mahalanobis(sc.transform(view.X))
        out = np.zeros(len(view.y))
        for f in np.unique(view.fabric):
            if f not in self.models:
                raise NotApplicable(f"no model trained for fabric {f}")
            sc, lw = self.models[f]
            sel = view.fabric == f
            out[sel] = lw.mahalanobis(sc.transform(view.X[sel]))
        return out


class IsolationForestScorer(_AnomalyBase):
    name = "M2 isolation forest (global)"

    def fit(self, train, val):
        normal = self._normal(train)
        self.scaler = StandardScaler().fit(train.X[normal])
        self.model = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=1).fit(
            self.scaler.transform(train.X[normal])
        )

    def score(self, view):
        return -self.model.score_samples(self.scaler.transform(view.X))


class OneClassSVMScorer(_AnomalyBase):
    name = "M2 one-class SVM (global)"

    def fit(self, train, val):
        normal = self._normal(train)
        self.scaler = StandardScaler().fit(train.X[normal])
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(
            self.scaler.transform(train.X[normal])
        )

    def score(self, view):
        return -self.model.decision_function(self.scaler.transform(view.X))


def method_factories() -> list[tuple[str, type | object]]:
    """(name, zero-argument constructor) for every method, in reporting order."""
    makers = [
        MajorityClass,
        RawContrast,
        RawBrightnessDeviation,
        RandomForest,
        SVM,
        lambda: Mahalanobis(per_fabric=False),
        lambda: Mahalanobis(per_fabric=True),
        IsolationForestScorer,
        OneClassSVMScorer,
    ]
    return [(m().name, m) for m in makers]
