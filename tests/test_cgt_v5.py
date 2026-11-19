"""
Unit tests for CGT-v5 Calibrated Controller
"""

import pytest
import numpy as np
from src.controllers.cgt_v5 import CGTv5CalibratedController


def test_cgt_v5_initialization():
    ctrl = CGTv5CalibratedController(tau_gate=0.30, max_rho=0.20, n_heavy_trees=10)
    assert ctrl.label == "post-hoc: developed after viewing test results"
    assert ctrl.tau_gate == 0.30
    assert ctrl.max_rho == 0.20
    assert ctrl.total_steps == 0
    assert ctrl.gate_open_steps == 0


def test_cgt_v5_predict_and_learn_loop():
    ctrl = CGTv5CalibratedController(tau_gate=0.30, max_rho=0.20, n_heavy_trees=5)
    rng = np.random.RandomState(42)

    for i in range(100):
        x = {f"feat_{j}": float(rng.randn()) for j in range(5)}
        y = int(x["feat_0"] > 0)
        
        pred = ctrl.predict_one(x)
        assert pred in (0, 1)
        ctrl.learn_one(x, y, y_pred=pred)
        
    assert ctrl.total_steps == 100
    assert 0.0 <= ctrl.duty_cycle <= 1.0


def test_cgt_v5_budget_enforcement():
    # Force max_rho to 0.05
    ctrl = CGTv5CalibratedController(tau_gate=0.99, max_rho=0.05, n_heavy_trees=3)
    rng = np.random.RandomState(42)

    for i in range(200):
        x = {f"feat_{j}": float(rng.randn()) for j in range(5)}
        y = int(x["feat_0"] > 0)
        pred = ctrl.predict_one(x)
        ctrl.learn_one(x, y, y_pred=pred)
        
    # Duty cycle should not significantly exceed max_rho
    # Allow 1-2 step slack due to discrete arrivals
    assert ctrl.duty_cycle <= 0.065
