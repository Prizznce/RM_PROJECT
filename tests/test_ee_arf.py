"""Unit tests for the EE-ARF-10 / conformal-gate upgrade."""

import json
import os
import random

import numpy as np
import pytest
from river.forest import ARFClassifier

from src.controllers.conformal_gate import (
    DriftAdaptiveConformalGate,
    StaticMarginGate,
    asymmetric_cost,
)
from src.controllers.ee_arf_controller import DEFAULT_ARF_PARAMS, AlwaysARF, EEARF10Controller
from src.eval.ee_harness import run_prequential
from src.eval.ee_metrics import block_metric, kappa_temporal
from src.models.ee_arf import EarlyExitARFClassifier
from src.stats.block_tests import holm


def _synthetic_stream(n=600, seed=0, drift_at=None):
    rng = random.Random(seed)
    out = []
    for t in range(n):
        x = {f"f{j}": rng.gauss(0, 1) for j in range(6)}
        s = x["f0"] + 0.5 * x["f1"]
        if drift_at is not None and t >= drift_at:
            s = -s
        out.append((x, int(s + 0.3 * rng.gauss(0, 1) > 0)))
    return out


# ---------------------------------------------------------------------- #
# Early-exit forest
# ---------------------------------------------------------------------- #
def test_subset_methods_reproduce_stock_arf_exactly():
    """learn_subset/predict_proba_subset over all trees == stock ARFClassifier."""
    stream = _synthetic_stream(500, drift_at=250)
    stock = ARFClassifier(n_models=10, seed=7, **DEFAULT_ARF_PARAMS)
    ee = EarlyExitARFClassifier(n_models=10, seed=7, **DEFAULT_ARF_PARAMS)
    allidx = range(10)
    for x, y in stream:
        assert stock.predict_proba_one(x) == ee.predict_proba_subset(x, allidx)
        stock.learn_one(x, y)
        ee.learn_subset(x, y, allidx)


def test_cached_core_votes_plus_expansion_is_bit_identical():
    stream = _synthetic_stream(400, drift_at=200)
    ee = EarlyExitARFClassifier(n_models=10, seed=3, **DEFAULT_ARF_PARAMS)
    for x, y in stream:
        votes = ee.weighted_votes(x, range(3))
        ee.weighted_votes(x, range(3, 10), into=votes)
        assert ee.normalise(votes) == ee.predict_proba_subset(x, range(10))
        ee.learn_subset(x, y, range(10))


def test_lockstep_matches_standalone_runs():
    from src.eval.ee_harness import run_prequential_lockstep

    stream = _synthetic_stream(300, drift_at=150)
    mk = lambda: [AlwaysARF(3, seed=1), EEARF10Controller(p_bg=0.2, seed=1),
                  EEARF10Controller(staleness="replay", seed=1)]
    lock = run_prequential_lockstep(mk(), stream, 0)
    for m, r in zip(mk(), lock):
        solo = run_prequential(m, stream, 0)
        assert np.array_equal(solo["_arrays"]["y_pred"], r["_arrays"]["y_pred"])
        assert solo["kappa_temporal"] == r["kappa_temporal"]


def test_core_subset_only_touches_core_trees():
    stream = _synthetic_stream(300)
    ee = EarlyExitARFClassifier(n_models=10, seed=1, **DEFAULT_ARF_PARAMS)
    ee.predict_proba_subset(stream[0][0], range(3))  # init ensemble
    for x, y in stream:
        ee.learn_subset(x, y, range(3))
    x0 = stream[0][0]
    assert all(ee.data[i].predict_proba_one(x0) for i in range(3))
    # trees 4..10 never saw a sample -> empty class distribution
    assert all(not ee.data[i].predict_proba_one(x0) for i in range(3, 10))
    assert all(ee._metrics[i].get() == 0.0 for i in range(3, 10))


# ---------------------------------------------------------------------- #
# Conformal gate
# ---------------------------------------------------------------------- #
def test_asymmetric_cost_normalised():
    assert asymmetric_cost(0, 0, 10) == 0.0
    assert asymmetric_cost(0, 1, 10) == 1.0          # FN
    assert asymmetric_cost(1, 0, 10) == pytest.approx(0.1)  # FP
    assert asymmetric_cost(1, 0, 1) == 1.0


def test_tau_update_direction_and_clipping():
    g = DriftAdaptiveConformalGate(alpha=0.1, gamma=0.5, tau_init=0.5, loss_mode="final_error")
    g.update(False, 0, 0, 1)  # loss 1 > alpha -> tau up
    assert g.tau == pytest.approx(0.95)
    g.update(False, 0, 0, 1)
    assert g.tau == 1.0  # clipped
    for _ in range(10):
        g.update(False, 1, 1, 1)  # loss 0 < alpha -> tau down
    assert 0.0 <= g.tau < 1.0


def test_exit_error_loss_is_zero_when_gated():
    g = DriftAdaptiveConformalGate(alpha=0.05, gamma=0.1, tau_init=0.3, loss_mode="exit_error")
    assert g.update(True, 0, 0, 1) == 0.0
    assert g.update(False, 0, 0, 1) == 1.0


def test_exit_error_long_run_risk_tracks_alpha():
    """With exit_error loss, the realised un-escalated error rate converges to alpha."""
    rng = np.random.RandomState(0)
    g = DriftAdaptiveConformalGate(alpha=0.05, gamma=0.02, tau_init=0.0, loss_mode="exit_error")
    T = 20000
    for _ in range(T):
        m = rng.rand()
        core_wrong = rng.rand() < (1 - m) * 0.4  # low margin => more errors
        gated = g.is_uncertain(m)
        g.update(gated, 0, 0 if core_wrong else 1, 1)
    assert abs(g.mean_loss - 0.05) < 0.01


def test_static_gate_does_not_move():
    g = StaticMarginGate(tau=0.4)
    for _ in range(50):
        g.update(False, 0, 0, 1)
    assert g.tau == 0.4


# ---------------------------------------------------------------------- #
# Controller
# ---------------------------------------------------------------------- #
def test_controller_trees_and_duty_accounting():
    stream = _synthetic_stream(400, drift_at=200)
    m = EEARF10Controller(alpha=0.05, gamma=0.05, p_bg=0.1, seed=3)
    res = run_prequential(m, stream, y_prev0=0)
    a = res["_arrays"]
    assert set(np.unique(a["trees_eval"])) <= {3, 10}
    rho = a["gated"].mean()
    assert res["avg_trees_eval"] == pytest.approx(3 + 7 * rho)
    assert res["duty_rho_pct"] == pytest.approx(100 * rho)
    assert np.all((a["tau"] >= 0) & (a["tau"] <= 1))


def test_controller_is_deterministic():
    stream = _synthetic_stream(300, drift_at=150)
    r1 = run_prequential(EEARF10Controller(p_bg=0.2, seed=11), stream, 0)
    r2 = run_prequential(EEARF10Controller(p_bg=0.2, seed=11), stream, 0)
    assert np.array_equal(r1["_arrays"]["y_pred"], r2["_arrays"]["y_pred"])
    assert np.array_equal(r1["_arrays"]["gated"], r2["_arrays"]["gated"])


def test_replay_ablation_has_no_background_updates_and_clears_buffer():
    stream = _synthetic_stream(600, drift_at=300)
    m = EEARF10Controller(staleness="replay", replay_buffer=200, seed=5)
    run_prequential(m, stream, 0)
    assert m.bg_updates == 0
    assert len(m.buffer) <= 200
    if m.total_alarms:
        assert m.replayed_samples > 0


def test_always_arf_trees_and_rho():
    stream = _synthetic_stream(100)
    r3 = run_prequential(AlwaysARF(3), stream, 0)
    r10 = run_prequential(AlwaysARF(10), stream, 0)
    assert r3["avg_trees_eval"] == 3 and r3["duty_rho_pct"] == 0
    assert r10["avg_trees_eval"] == 10 and r10["duty_rho_pct"] == 100


def test_prediction_never_sees_current_label():
    """Model whose predict_one would leak y_t via a side channel must not score 100%."""
    stream = _synthetic_stream(200, seed=9)
    m = EEARF10Controller(seed=1)
    orig_learn = m.learn_one
    seen = []

    def spy(x, y, y_pred=None):
        seen.append(y)
        return orig_learn(x, y, y_pred)

    m.learn_one = spy
    preds = []
    orig_predict = m.predict_one

    def pspy(x):
        preds.append(len(seen))  # number of labels revealed before this prediction
        return orig_predict(x)

    m.predict_one = pspy
    run_prequential(m, stream, 0)
    assert preds == list(range(len(stream)))


# ---------------------------------------------------------------------- #
# Metrics / stats
# ---------------------------------------------------------------------- #
def test_kappa_temporal_persistence_is_zero():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0])
    per = np.concatenate([[0], y[:-1]])
    assert kappa_temporal(y, per, 0) == pytest.approx(0.0)
    assert kappa_temporal(y, y, 0) == pytest.approx(1.0)


def test_block_metric_shapes():
    y = np.random.RandomState(0).randint(0, 2, 1000)
    assert block_metric(y, y, 168, "accuracy", 0).shape == (5,)


def test_holm_monotone():
    adj = holm([0.01, 0.04, 0.03, 0.2])
    assert np.all(adj >= np.array([0.01, 0.04, 0.03, 0.2]))
    assert adj[0] == pytest.approx(0.04)


# ---------------------------------------------------------------------- #
# Frozen config integrity (only once the experiment has been frozen)
# ---------------------------------------------------------------------- #
CFG = "configs/ee_arf_frozen_config.json"


@pytest.mark.skipif(not os.path.exists(CFG), reason="EE-ARF config not frozen yet")
def test_ee_arf_frozen_config_hash():
    from scripts.run_ee_arf_experiment import canonical_hash

    with open(CFG, encoding="utf-8") as f:
        data = json.load(f)
    recorded = data.pop("sha256")
    assert canonical_hash(data) == recorded


# ---------------------------------------------------------------------- #
# Legacy adapters must not change legacy predictions
# ---------------------------------------------------------------------- #
def test_legacy_adapters_reproduce_raw_controllers():
    from src.controllers.cgt_v3 import CGTv3LearnedGateController
    from src.controllers.cgt_v4 import CGTv4Controller
    from src.controllers.legacy_adapters import ARF3Wrapper, CGTv3Adapter, CGTv4Adapter, OMSMABAdapter
    from src.controllers.oms_mab import OMSMABController

    stream = _synthetic_stream(500, drift_at=250)
    v3cfg = {"tau_gate": 0.25, "max_rho": 0.12, "h_cooldown": 50, "buffer_b": 200, "adwin_delta": 0.005}
    arfcfg = {"grace_period": 200, "lambda_value": 6, "delta": 0.001, "split_criterion": "info_gain"}
    v4cfg = {"tau_gate": 0.25, "max_rho": 0.12, "h_cooldown": 50, "replay_mode": "amortized",
             "buffer_b": 50, "k_replay_per_step": 5, "batch_size": 16, "n_heavy_trees": 3}

    raws = [
        OMSMABController(seed=1),
        CGTv3LearnedGateController(arf_model=ARF3Wrapper(seed=1), seed=1, **v3cfg),
        CGTv4Controller(seed=1, **v4cfg),
    ]
    adapted = [OMSMABAdapter(seed=1), CGTv3Adapter(v3cfg, arfcfg, seed=1), CGTv4Adapter(v4cfg, seed=1)]
    for raw, ad in zip(raws, adapted):
        res = run_prequential(ad, stream, 0)
        preds = []
        for x, y in stream:
            p = raw.predict_one(x)
            p = 0 if p is None else int(p)
            raw.learn_one(x, y, p)
            preds.append(p)
        assert np.array_equal(np.array(preds), res["_arrays"]["y_pred"]), ad.name
        assert np.all((res["_arrays"]["p1"] >= 0) & (res["_arrays"]["p1"] <= 1))
        # deployed P(1) agrees with the deployed label (except exact ties)
        p1, yp = res["_arrays"]["p1"], res["_arrays"]["y_pred"]
        assert np.mean((p1 > 0.5) == (yp == 1)) > 0.97, ad.name
