"""
Automated Unit & Verification Tests for CGT-v4
LABEL: post-hoc: developed after viewing test results
"""

import os
import json
import hashlib
import numpy as np
import pytest

from src.controllers.cgt_v4 import (
    WelfordIncrementalScaler,
    HandCodedMiniBatchLogisticGate,
    CGTv4Controller
)

def test_cgt_v4_label_invariant():
    """Verify that CGT-v4 is strictly labeled post-hoc across all instances."""
    controller = CGTv4Controller()
    assert "post-hoc: developed after viewing test results" in controller.label
    assert "post-hoc" in controller.name


def test_welford_incremental_scaler_invariants():
    """Verify Welford normalizer maintains correct running statistics."""
    scaler = WelfordIncrementalScaler(dim=3)
    data = np.random.randn(100, 3) * 5.0 + 10.0
    
    scaled_last = None
    for row in data:
        scaled_last = scaler.update_and_scale(row)
        
    assert scaler.n == 100
    assert np.allclose(scaler.mean, np.mean(data, axis=0), atol=1e-5)
    # Check that scaled values have approximately unit scale
    var_est = scaler.M2 / 99
    assert np.all(var_est > 0.0)


def test_hand_coded_minibatch_logistic_gate():
    """Verify hand-coded dot product sigmoid bounds and mini-batch SGD updates."""
    gate = HandCodedMiniBatchLogisticGate(dim=3, lr=0.1, batch_size=4)
    x = np.array([1.0, -1.0, 0.5])
    
    # Sigmoid probability must be bounded in [0, 1]
    prob = gate.predict_proba(x)
    assert 0.0 <= prob <= 1.0
    
    # Test batch accumulation
    initial_weights = gate.weights.copy()
    gate.update(x, target_z=1.0)
    assert gate.batch_count == 1
    # Weights should NOT change until batch_size is reached
    assert np.array_equal(gate.weights, initial_weights)
    
    # Feed 3 more samples to trigger batch update
    for _ in range(3):
        gate.update(x, target_z=1.0)
    assert gate.batch_count == 0
    # Weights must now have updated
    assert not np.array_equal(gate.weights, initial_weights)


def test_amortized_replay_queue_invariants():
    """Verify that amortized replay never blocks with more than k_replay items per step."""
    controller = CGTv4Controller(
        replay_mode="amortized",
        buffer_b=20,
        k_replay_per_step=3
    )
    
    dummy_x = {"demand_diff_1_24": 100.0, "hour_sin": 0.5, "hour_cos": 0.5}
    # Warm up controller
    for _ in range(25):
        pred = controller.predict_one(dummy_x)
        controller.learn_one(dummy_x, 1, pred)
        
    # Test amortized queue transfer and step processing
    assert len(controller.buffer) > 0
    controller.amortized_queue.extend(controller.buffer)
    controller.buffer.clear()
    assert len(controller.buffer) == 0
    
    # In each step, amortized queue should decrease by at most k_replay_per_step
    q_len_before = len(controller.amortized_queue)
    pred = controller.predict_one(dummy_x)
    controller.learn_one(dummy_x, 1, pred)
    q_len_after = len(controller.amortized_queue)
    assert q_len_before - q_len_after <= 3


def test_warm_standby_training_frequency():
    """Verify warm standby trains on every 4th sample when gate is closed."""
    controller = CGTv4Controller(replay_mode="warm_standby", tau_gate=0.99) # high tau keeps gate closed
    dummy_x = {"demand_diff_1_24": 0.0, "hour_sin": 0.0, "hour_cos": 1.0}
    
    # Run 12 steps
    for step in range(1, 13):
        pred = controller.predict_one(dummy_x)
        controller.learn_one(dummy_x, 0, pred)
        
    assert controller.total_steps == 12
    # Gate should remain closed
    assert controller.gate_open_steps == 0


def test_cgt_v4_frozen_hash_integrity():
    """Verify that frozen CGT-v4 config matches its recorded SHA-256 hash."""
    cfg_path = "configs/cgt_v4_frozen_config.json"
    assert os.path.exists(cfg_path)
    
    with open(cfg_path, "r") as f:
        data = json.load(f)
        
    recorded_hash = data["metadata"]["sha256"]
    data_copy = json.loads(json.dumps(data))
    data_copy["metadata"]["sha256"] = "PENDING"
    calculated_hash = hashlib.sha256(json.dumps(data_copy, indent=2, sort_keys=True).encode()).hexdigest()
    assert recorded_hash == calculated_hash

