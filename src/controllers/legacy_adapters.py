"""
Adapters that run the *unmodified* legacy controllers (OMS-MAB, CGT-v3, CGT-v4)
inside the EE-ARF lockstep harness.

The harness needs, per step: the deployed P(y=1) (for PR-AUC), whether the
heavy tier was invoked, trees evaluated on the decision path, and trees
updated. The legacy controllers do not expose these, so their base learners
are wrapped in ``_ProbaProxy`` objects that

  * reproduce river's ``predict_one`` exactly (argmax of predict_proba_one,
    None/0 when empty) from a single predict_proba_one call - no extra tree
    evaluations, identical predictions;
  * cache the last P(y=1);
  * count learn_one calls (to report trees updated per step).

Controller logic is untouched.
"""

from __future__ import annotations

from river.forest import ARFClassifier

from src.controllers.cgt_v3 import CGTv3LearnedGateController
from src.controllers.cgt_v4 import CGTv4Controller
from src.controllers.oms_mab import OMSMABController
from src.models.base_models import HoeffdingTreeWrapper


def _p1(proba: dict) -> float:
    if not proba:
        return 0.5
    tot = sum(proba.values())
    return float(proba.get(1, 0.0)) / tot if tot > 0 else 0.5


class _ProbaProxy:
    def __init__(self, inner, river_model, n_trees: int, none_value):
        self._inner = inner
        self._river = river_model
        self.n_trees = n_trees
        self._none = none_value
        self.last_p1 = 0.5
        self.n_learn = 0

    def predict_one(self, x):
        proba = self._river.predict_proba_one(x)
        self.last_p1 = _p1(proba)
        if not proba:
            return self._none
        y = max(proba, key=proba.get)
        return y if self._none is None else int(y)

    def learn_one(self, x, y, *args, **kwargs):
        self.n_learn += 1
        return self._inner.learn_one(x, y)

    def __getattr__(self, item):  # e.g. ``.model`` used by legacy code
        return getattr(self._inner, item)


class ARF3Wrapper:
    """Same as scripts/run_fair_comparison_arf3_vs_cgt.py::ARF3Wrapper."""

    def __init__(self, grace_period=200, lambda_value=6, delta=0.001, split_criterion="info_gain", seed=42):
        self.model = ARFClassifier(n_models=3, grace_period=grace_period, lambda_value=lambda_value,
                                   delta=delta, split_criterion=split_criterion, seed=seed)

    def predict_one(self, x):
        p = self.model.predict_one(x)
        return 0 if p is None else int(p)

    def learn_one(self, x, y, y_pred=None):
        self.model.learn_one(x, y)


class _Adapter:
    heavy_trees = 0

    def __init__(self, name):
        self.name = name
        self.last_p1 = 0.5
        self.last_gated = False
        self.last_trees_eval = 1
        self.last_trees_trained = 0

    def _learn_count(self):
        return sum(p.n_learn * p.n_trees for p in self._proxies)

    def learn_one(self, x, y, y_pred=None):
        before = self._learn_count()
        self.ctrl.learn_one(x, y, y_pred)
        self.last_trees_trained = self._learn_count() - before

    def diagnostics(self):
        return {"adwin_alarms": getattr(self.ctrl, "total_alarms", getattr(self.ctrl, "total_drifts_detected", 0))}


class OMSMABAdapter(_Adapter):
    """OMS-MAB (Wilson et al., 2024) as implemented in src/controllers/oms_mab.py:
    D-UCB over {HT, ARF-10 (repo ARFFullWrapper config)}, HDDM_A per arm,
    selected-arm-only training."""

    def __init__(self, seed=42, gamma=0.98, c=0.5):
        super().__init__("OMS-MAB (Wilson et al. 2024)")
        self.ctrl = OMSMABController(gamma=gamma, c=c, seed=seed)
        ht, arf = self.ctrl.arms
        self._proxies = [_ProbaProxy(ht, ht.model, 1, 0), _ProbaProxy(arf, arf.model, arf.n_models, 0)]
        self.ctrl.arms = self._proxies

    def predict_one(self, x):
        y = self.ctrl.predict_one(x)
        idx = self.ctrl.active_arm_idx
        self.last_gated = idx == 1
        self.last_trees_eval = self._proxies[idx].n_trees
        self.last_p1 = self._proxies[idx].last_p1
        return int(y)


class CGTv3Adapter(_Adapter):
    """CGT-v3 learned meta-gate, HT fast path, ARF-3 heavy arm, B=200 blocking replay."""

    def __init__(self, cfg: dict, arf_cfg: dict, seed=42):
        super().__init__("CGT-v3 (Learned Gate, ARF-3, Replay B=200)")
        arf = ARF3Wrapper(grace_period=arf_cfg["grace_period"], lambda_value=arf_cfg["lambda_value"],
                          delta=arf_cfg["delta"], split_criterion=arf_cfg["split_criterion"], seed=seed)
        self.ctrl = CGTv3LearnedGateController(
            tau_gate=cfg["tau_gate"], max_rho=cfg["max_rho"], h_cooldown=cfg["h_cooldown"],
            buffer_b=cfg["buffer_b"], adwin_delta=cfg["adwin_delta"], arf_model=arf, seed=seed,
        )
        ht = self.ctrl.ht
        self._ht = _ProbaProxy(ht, ht.model, 1, 0)
        self._arf = _ProbaProxy(arf, arf.model, 3, 0)
        self.ctrl.ht, self.ctrl.arf = self._ht, self._arf
        self._proxies = [self._ht, self._arf]

    def predict_one(self, x):
        y = self.ctrl.predict_one(x)
        self.last_gated = bool(self.ctrl.gate_open)
        self.last_trees_eval = 1 + (3 if self.last_gated else 0)
        self.last_p1 = (self._arf if self.last_gated else self._ht).last_p1
        return int(y)


class CGTv4Adapter(_Adapter):
    """CGT-v4 (amortized replay, hand-coded mini-batch gate, Welford scaler, ARF-3)."""

    def __init__(self, cfg: dict, seed=42):
        super().__init__("CGT-v4 (Amortized Replay, ARF-3)")
        self.ctrl = CGTv4Controller(
            tau_gate=cfg["tau_gate"], max_rho=cfg["max_rho"], h_cooldown=cfg["h_cooldown"],
            replay_mode=cfg["replay_mode"], buffer_b=cfg["buffer_b"],
            k_replay_per_step=cfg["k_replay_per_step"], batch_size=cfg["batch_size"],
            n_heavy_trees=cfg["n_heavy_trees"], seed=seed,
        )
        self._n_heavy = cfg["n_heavy_trees"]
        # HT is accessed via .ht.model.predict_proba_one internally; only count its learns
        self._ht = _ProbaProxy(self.ctrl.ht, self.ctrl.ht.model, 1, 0)
        self.ctrl.ht = self._ht
        arf = self.ctrl.arf
        self._arf = _ProbaProxy(arf, arf, self._n_heavy, None)
        self.ctrl.arf = self._arf
        self._proxies = [self._ht, self._arf]

    def predict_one(self, x):
        y = self.ctrl.predict_one(x)
        self.last_gated = bool(self.ctrl.gate_open)
        self.last_trees_eval = 1 + (self._n_heavy if self.last_gated else 0)
        if self.last_gated:
            self.last_p1 = self._arf.last_p1
        else:  # reconstruct HT P(1) from the controller's cached margin & label
            m = self.ctrl._cached_margin
            self.last_p1 = (1 + m) / 2 if self.ctrl._cached_ht_pred == 1 else (1 - m) / 2
        return int(y)
