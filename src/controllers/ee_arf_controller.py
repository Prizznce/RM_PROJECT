"""
EE-ARF-10: Early-Exit Adaptive Random Forest controller + uniform baseline wrappers.

All models expose the same step interface used by ``src/eval/ee_harness.py``:

    y_hat = model.predict_one(x)    # sets: last_p1, last_trees_eval, last_gated
    model.learn_one(x, y)           # sets: last_trees_trained

Per-step flow of EE-ARF-10 (strict test-then-train)
---------------------------------------------------
predict_one(x_t):
    p_core   = weighted vote of trees 1..3            (always)
    m_t      = |p_core(1) - 0.5| * 2
    g_t      = (m_t < tau_t) or drift_flag_active
    if g_t:  p = weighted vote of trees 1..10         (expansion)
    else:    p = p_core                               (early exit)
learn_one(x_t, y_t):
    gate.update(...)                                  (tau_{t+1}, uses y_t)
    ADWIN.update(final error) -> may raise drift flag for `drift_hold` steps
    trees 1..3 learn                                  (always)
    trees 4..10 learn if g_t or U(0,1) < p_bg          (staleness = "background")
       -- or --
    trees 4..10 learn if g_t; non-gated samples go to a B-sized FIFO that is
    replayed into trees 4..10 when ADWIN fires        (staleness = "replay")
"""

from __future__ import annotations

import random
from collections import deque

from river.drift import ADWIN
from river.forest import ARFClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier, HoeffdingTreeClassifier

from src.controllers.conformal_gate import DriftAdaptiveConformalGate, StaticMarginGate
from src.models.ee_arf import EarlyExitARFClassifier

# Shared tree hyper-parameters for every ARF variant (validation-tuned ARF-3
# config from configs/fair_comparison_frozen_configs.json).
DEFAULT_ARF_PARAMS = {
    "grace_period": 200,
    "lambda_value": 6,
    "delta": 0.001,
    "split_criterion": "info_gain",
}
# Existing Always-HT configuration (src/models/base_models.py).
DEFAULT_HT_PARAMS = {"grace_period": 200, "delta": 1e-7}


def _p1(proba: dict) -> float:
    if not proba:
        return 0.5
    tot = sum(proba.values())
    return float(proba.get(1, 0.0)) / tot if tot > 0 else 0.5


def _label(p1: float) -> int:
    return 1 if p1 > 0.5 else 0


# ====================================================================== #
# Proposed controller
# ====================================================================== #
class EEARF10Controller:
    def __init__(
        self,
        name: str = "EE-ARF-10",
        n_trees: int = 10,
        n_core: int = 3,
        gate: str = "conformal",           # "conformal" | "static"
        staleness: str = "background",     # "background" | "replay"
        alpha: float = 0.05,
        gamma: float = 0.01,
        tau_init: float = 0.30,
        tau_static: float = 0.30,
        w_fn: float = 1.0,
        loss_mode: str = "exit_error",
        p_bg: float = 0.10,
        replay_buffer: int = 200,
        use_drift_flag: bool = True,
        adwin_delta: float = 0.002,
        drift_hold: int = 50,
        arf_params: dict | None = None,
        seed: int = 42,
    ):
        if gate not in ("conformal", "static"):
            raise ValueError("gate must be 'conformal' or 'static'")
        if staleness not in ("background", "replay"):
            raise ValueError("staleness must be 'background' or 'replay'")
        if not 0 < n_core < n_trees:
            raise ValueError("need 0 < n_core < n_trees")

        self.name = name
        self.n_trees = n_trees
        self.n_core = n_core
        self.core_idx = tuple(range(n_core))
        self.ext_idx = tuple(range(n_core, n_trees))
        self.all_idx = tuple(range(n_trees))
        self.staleness = staleness
        self.p_bg = float(p_bg) if staleness == "background" else 0.0
        self.use_drift_flag = use_drift_flag
        self.drift_hold = int(drift_hold)

        params = dict(DEFAULT_ARF_PARAMS)
        params.update(arf_params or {})
        self.forest = EarlyExitARFClassifier(n_models=n_trees, seed=seed, **params)

        if gate == "conformal":
            self.gate = DriftAdaptiveConformalGate(
                alpha=alpha, gamma=gamma, tau_init=tau_init, w_fn=w_fn, loss_mode=loss_mode
            )
        else:
            self.gate = StaticMarginGate(tau=tau_static, w_fn=w_fn)

        self.detector = ADWIN(delta=adwin_delta)
        self._bg_rng = random.Random(seed + 7919)  # independent of forest RNG
        self.buffer = deque(maxlen=replay_buffer) if staleness == "replay" else None

        # step state
        self.drift_counter = 0
        self.last_p1 = 0.5
        self.last_margin = 0.0
        self.last_core_pred = 0
        self.last_pred = 0
        self.last_gated = False
        self.last_trees_eval = n_core
        self.last_trees_trained = 0
        self.last_tau = self.gate.tau
        self.last_gate_loss = 0.0

        # diagnostics
        self.total_steps = 0
        self.gated_steps = 0
        self.drift_gated_steps = 0
        self.total_alarms = 0
        self.replayed_samples = 0
        self.bg_updates = 0

    # ------------------------------------------------------------------ #
    def predict_one(self, x: dict) -> int:
        self.total_steps += 1
        self.last_tau = self.gate.tau

        core_votes = self.forest.weighted_votes(x, self.core_idx)
        p_core = _p1(self.forest.normalise(core_votes))
        margin = abs(p_core - 0.5) * 2.0
        core_pred = _label(p_core)

        drift_active = self.use_drift_flag and self.drift_counter > 0
        gated = self.gate.is_uncertain(margin) or drift_active
        if drift_active:
            self.drift_counter -= 1
            if not self.gate.is_uncertain(margin):
                self.drift_gated_steps += 1

        if gated:
            # expansion: evaluate only trees 4..10 and add to the cached core votes
            self.forest.weighted_votes(x, self.ext_idx, into=core_votes)
            p = _p1(self.forest.normalise(core_votes))
            self.gated_steps += 1
            self.last_trees_eval = self.n_trees
        else:
            p = p_core
            self.last_trees_eval = self.n_core

        self.last_margin = margin
        self.last_core_pred = core_pred
        self.last_gated = gated
        self.last_p1 = p
        self.last_pred = _label(p)
        return self.last_pred

    # ------------------------------------------------------------------ #
    def learn_one(self, x: dict, y: int, y_pred: int | None = None) -> None:
        y_hat = self.last_pred if y_pred is None else y_pred

        # 1. Conformal threshold update (uses revealed label only)
        self.last_gate_loss = self.gate.update(self.last_gated, y_hat, self.last_core_pred, y)

        # 2. Drift detector on the deployed prediction's error
        self.detector.update(int(y_hat != y))
        alarm = self.detector.drift_detected
        if alarm:
            self.total_alarms += 1
            if self.use_drift_flag:
                self.drift_counter = self.drift_hold

        trained = 0
        # 3. Core trees always learn
        self.forest.learn_subset(x, y, self.core_idx)
        trained += self.n_core

        # 4. Expansion trees
        if self.staleness == "replay":
            if alarm and self.buffer:
                for bx, by in self.buffer:
                    self.forest.learn_subset(bx, by, self.ext_idx)
                trained += len(self.ext_idx) * len(self.buffer)
                self.replayed_samples += len(self.buffer)
                self.buffer.clear()
            if self.last_gated:
                self.forest.learn_subset(x, y, self.ext_idx)
                trained += len(self.ext_idx)
            else:
                self.buffer.append((x, y))
        else:  # background
            if self.last_gated or (self.p_bg > 0.0 and self._bg_rng.random() < self.p_bg):
                if not self.last_gated:
                    self.bg_updates += 1
                self.forest.learn_subset(x, y, self.ext_idx)
                trained += len(self.ext_idx)

        self.last_trees_trained = trained

    # ------------------------------------------------------------------ #
    @property
    def duty_cycle(self) -> float:
        return self.gated_steps / max(self.total_steps, 1)

    def diagnostics(self) -> dict:
        return {
            "final_tau": round(self.gate.tau, 4),
            "gate_mean_loss": round(self.gate.mean_loss, 4),
            "adwin_alarms": self.total_alarms,
            "drift_forced_steps": self.drift_gated_steps,
            "replayed_samples": self.replayed_samples,
            "bg_updates": self.bg_updates,
        }


# ====================================================================== #
# Uniform baselines
# ====================================================================== #
class AlwaysTree:
    """Always-HT / Always-HAT."""

    def __init__(self, kind: str = "HT", seed: int = 42, **params):
        p = dict(DEFAULT_HT_PARAMS)
        p.update(params)
        if kind == "HT":
            self.model = HoeffdingTreeClassifier(**p)
        elif kind == "HAT":
            self.model = HoeffdingAdaptiveTreeClassifier(seed=seed, **p)
        else:
            raise ValueError(kind)
        self.name = f"Always-{kind}"
        self.last_p1 = 0.5
        self.last_gated = False
        self.last_trees_eval = 1
        self.last_trees_trained = 1

    def predict_one(self, x):
        self.last_p1 = _p1(self.model.predict_proba_one(x))
        return _label(self.last_p1)

    def learn_one(self, x, y, y_pred=None):
        self.model.learn_one(x, y)

    def diagnostics(self):
        return {}


class AlwaysARF:
    """Always-ARF-n using stock river ARFClassifier (100% duty)."""

    def __init__(self, n_models: int, seed: int = 42, arf_params: dict | None = None):
        params = dict(DEFAULT_ARF_PARAMS)
        params.update(arf_params or {})
        self.model = ARFClassifier(n_models=n_models, seed=seed, **params)
        self.n_models = n_models
        self.name = f"Always-ARF-{n_models}"
        self.last_p1 = 0.5
        # rho = share of samples that evaluate trees 4..10 -> only ARF-n with n > 3
        self.last_gated = n_models > 3
        self.last_trees_eval = n_models
        self.last_trees_trained = n_models

    def predict_one(self, x):
        self.last_p1 = _p1(self.model.predict_proba_one(x))
        return _label(self.last_p1)

    def learn_one(self, x, y, y_pred=None):
        self.model.learn_one(x, y)

    def diagnostics(self):
        return {}
