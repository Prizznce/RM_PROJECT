"""
Early-Exit Adaptive Random Forest (EE-ARF).

Thin subclass of river's ``ARFClassifier`` that adds *subset* inference and
*subset* training, so a single 10-tree forest can be evaluated / updated on
trees 1-3 only (the always-on core) or on all 10 trees (the expansion).

The two subset methods mirror ``ARFClassifier.predict_proba_one`` and
``BaseForest.learn_one`` (river 0.27) line for line, restricted to an index
set. Consequently, calling them with ``indices = range(n_models)`` reproduces
stock ARF exactly (verified in tests/test_ee_arf.py), which is what makes the
Always-ARF-n baselines and EE-ARF-10 directly comparable.
"""

from __future__ import annotations

import collections

from river import metrics
from river.forest import ARFClassifier
from river.utils.random import poisson


class EarlyExitARFClassifier(ARFClassifier):
    """ARF with ``predict_proba_subset`` / ``learn_subset``."""

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def weighted_votes(self, x: dict, indices, into: collections.Counter | None = None) -> collections.Counter:
        """Un-normalised accuracy-weighted vote mass of the trees in ``indices``.

        Passing the core's Counter as ``into`` lets the expansion accumulate
        trees 4..10 on top of it in exactly river's summation order, so
        votes(1..3) then += trees 4..10 is bit-identical to votes(1..10) and no
        tree is evaluated twice.
        """
        y_pred: collections.Counter = collections.Counter() if into is None else into
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))
            return y_pred

        for i in indices:
            y_proba_temp = self.data[i].predict_proba_one(x)
            metric_value = self._metrics[i].get()
            if not self.disable_weighted_vote and metric_value > 0.0:
                y_proba_temp = {k: v * metric_value for k, v in y_proba_temp.items()}
            y_pred.update(y_proba_temp)
        return y_pred

    @staticmethod
    def normalise(votes: collections.Counter) -> dict:
        total = sum(votes.values())
        if total > 0:
            return {label: p / total for label, p in votes.items()}
        return {}

    def predict_proba_subset(self, x: dict, indices) -> dict:
        """Accuracy-weighted soft vote over the trees in ``indices``."""
        return self.normalise(self.weighted_votes(x, indices))

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def learn_subset(self, x: dict, y, indices) -> None:
        """Online Poisson(lambda) bagging + per-tree ADWIN warning/drift
        handling, applied only to the trees in ``indices``."""
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))

        for i in indices:
            model = self.data[i]
            y_pred = model.predict_one(x)

            self._metrics[i].update(
                y_true=y,
                y_pred=(
                    model.predict_proba_one(x)
                    if isinstance(self.metric, metrics.base.ClassificationMetric)
                    and not self.metric.requires_labels
                    else y_pred
                ),
            )

            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k <= 0:
                continue

            if not self._warning_detection_disabled and self._background[i] is not None:
                self._background[i].learn_one(x=x, y=y, w=k)

            model.learn_one(x=x, y=y, w=k)

            drift_input = None
            if not self._warning_detection_disabled:
                drift_input = self._drift_detector_input(i, y, y_pred)
                self._warning_detectors[i].update(drift_input)
                if self._warning_detectors[i].drift_detected:
                    self._background[i] = self._new_base_model()
                    self._warning_detectors[i] = self.warning_detector.clone()
                    self._warning_tracker[i] += 1

            if not self._drift_detection_disabled:
                drift_input = (
                    drift_input
                    if drift_input is not None
                    else self._drift_detector_input(i, y, y_pred)
                )
                self._drift_detectors[i].update(drift_input)
                if self._drift_detectors[i].drift_detected:
                    if not self._warning_detection_disabled and self._background[i] is not None:
                        self.data[i] = self._background[i]
                        self._background[i] = None
                        self._warning_detectors[i] = self.warning_detector.clone()
                        self._drift_detectors[i] = self.drift_detector.clone()
                        self._metrics[i] = self.metric.clone()
                    else:
                        self.data[i] = self._new_base_model()
                        self._drift_detectors[i] = self.drift_detector.clone()
                        self._metrics[i] = self.metric.clone()
                    self._drift_tracker[i] += 1
