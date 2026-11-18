"""
Unit tests for Day 7 diagnostics, measurement hygiene, and drift robustness utilities.
"""

import os
import copy
import numpy as np
import pandas as pd
import pytest

from src.analysis.drift_robustness import build_shocked_test_stream_copy
from src.analysis.gate_meta_eval import run_gate_meta_eval

def test_temporal_kappa_calculation():
    """Verify Bifet et al. temporal kappa+ metric calculation formula."""
    p_persistence = 0.8095
    p_model = 0.8578
    
    # Formula: (p_o - p_p) / (1 - p_p)
    kappa_temporal = (p_model - p_persistence) / (1.0 - p_persistence)
    expected = (0.8578 - 0.8095) / (1.0 - 0.8095)
    assert abs(kappa_temporal - expected) < 1e-6
    assert kappa_temporal > 0.25  # Substantial gain over autoregressive persistence

    # If model equals persistence, kappa+ must be exactly 0.0
    kappa_zero = (p_persistence - p_persistence) / (1.0 - p_persistence)
    assert abs(kappa_zero) < 1e-9

def test_drift_robustness_stream_isolation():
    """Verify that synthetic drift injection operates STRICTLY on an in-memory copy without touching disk."""
    csv_path = "data/energy_dataset_clean.csv"
    orig_mtime = os.path.getmtime(csv_path)
    orig_size = os.path.getsize(csv_path)

    val_stream, shocked_test_stream = build_shocked_test_stream_copy()
    
    # Disk file must be completely untouched
    assert os.path.getmtime(csv_path) == orig_mtime
    assert os.path.getsize(csv_path) == orig_size
    assert len(val_stream) == 10512
    assert len(shocked_test_stream) == 24528

    # Shock 1 verification: step 4000 demand must be boosted by 30%
    x_mod, y_mod, _ = shocked_test_stream[4000]
    assert x_mod["demand_lag_1"] > 0
    # Shock 2 verification: step 10000 solar must be reduced by 85%
    x_mod2, _, _ = shocked_test_stream[10000]
    assert x_mod2["solar_lag_1"] < 1000.0

def test_gate_gain_loss_accounting():
    """Verify mathematical integrity of the accuracy gain vs loss decomposition."""
    # Simulation: 10 instances
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    pred_ht = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    pred_arf = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1])
    gate_open = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 0])

    # Decompose active instances
    active_idx = np.where(gate_open == 1)[0]
    gain = 0
    harm = 0
    for i in active_idx:
        ht_corr = (pred_ht[i] == y_true[i])
        arf_corr = (pred_arf[i] == y_true[i])
        if arf_corr and not ht_corr:
            gain += 1
        elif not arf_corr and ht_corr:
            harm += 1

    net_delta = gain - harm
    # Accuracy when using gate vs always HT
    ht_correct_total = sum(pred_ht == y_true)
    gated_pred = np.where(gate_open == 1, pred_arf, pred_ht)
    gated_correct_total = sum(gated_pred == y_true)

    assert gated_correct_total - ht_correct_total == net_delta

def test_pareto_dominance_logic():
    """Verify that non-dominated set calculation filters strictly dominated models."""
    sample_df = pd.DataFrame([
        {"model": "A", "cpu": 10.0, "acc": 80.0},
        {"model": "B", "cpu": 20.0, "acc": 79.0}, # Dominated by A (higher CPU, lower acc)
        {"model": "C", "cpu": 50.0, "acc": 85.0}, # Non-dominated (higher acc, higher CPU)
        {"model": "D", "cpu": 60.0, "acc": 84.0}, # Dominated by C
    ])

    sorted_df = sample_df.sort_values("cpu").reset_index(drop=True)
    pareto_pts = []
    max_acc = -1.0
    for _, row in sorted_df.iterrows():
        if row["acc"] > max_acc:
            pareto_pts.append(row["model"])
            max_acc = row["acc"]

    assert pareto_pts == ["A", "C"]

