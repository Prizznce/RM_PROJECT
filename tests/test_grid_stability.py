"""
Unit tests for Synchronous Grid Physics & Frequency Stability Simulation.
Verifies the Swing Equation, RoCoF calculation, and the latency-physics coupling.
"""

import pytest
import numpy as np
from src.physics.grid_stability import GridFrequencyDynamics, evaluate_models_on_grid_contingencies

def test_nominal_frequency_and_rocof_zero():
    """Confirms that when generation equals load, df/dt is exactly zero."""
    grid = GridFrequencyDynamics(f0=50.00)
    rocof = grid.compute_rocof(net_imbalance_mw=0.0)
    assert abs(rocof) < 1e-6, f"RoCoF should be zero at equilibrium, got {rocof}"

def test_uncontrolled_shock_triggers_ufls():
    """Confirms that a severe 3,500 MW shock without BESS breaches 49.50 Hz and trips UFLS."""
    grid = GridFrequencyDynamics(f0=50.00, ufls_stage1_hz=49.50)
    res = grid.simulate_trajectory(shock_magnitude_mw=3500.0, bess_enabled=False)
    assert res['ufls_tripped'] is True, "3,500 MW shock without BESS must trigger UFLS"
    assert res['f_nadir'] <= 49.50, f"Nadir should breach 49.50 Hz, got {res['f_nadir']}"
    assert res['blackout_prevented'] is False

def test_cgt_v3_sub_millisecond_prevents_blackout():
    """Confirms that CGT-v3 (< 1,000 us latency) enables fast BESS injection and preserves grid."""
    grid = GridFrequencyDynamics(f0=50.00, ufls_stage1_hz=49.50)
    res = grid.simulate_trajectory(
        shock_magnitude_mw=3500.0,
        bess_enabled=True,
        inference_latency_us=827.3,
        prediction_correct=True
    )
    assert res['blackout_prevented'] is True, "CGT-v3 must prevent blackout on 3,500 MW shock"
    assert res['f_nadir'] >= 49.60, f"Frequency nadir should stay safe (>=49.60 Hz), got {res['f_nadir']}"
    assert res['ufls_tripped'] is False

def test_deep_learning_latency_penalty_causes_trip():
    """Confirms that a 12,450 us latency (Deep Neural Net) causes fatal actuation delay."""
    grid = GridFrequencyDynamics(f0=50.00, ufls_stage1_hz=49.50)
    res = grid.simulate_trajectory(
        shock_magnitude_mw=3500.0,
        bess_enabled=True,
        inference_latency_us=12450.0,  # 12.45 ms
        prediction_correct=True
    )
    # The severe actuation delay causes frequency to drop below 49.50 Hz before BESS ramps
    assert res['f_nadir'] <= 49.50, f"Deep learning latency must breach 49.50 Hz, got {res['f_nadir']}"
    assert res['ufls_tripped'] is True, "Deep learning latency must trigger UFLS feeder trip"

def test_stochastic_contingency_suite():
    """Tests the 100-contingency monte-carlo simulation function."""
    df = evaluate_models_on_grid_contingencies(n_contingencies=15, seed=42)
    assert len(df) == 6, f"Expected 6 model architectures, got {len(df)}"
    assert 'blackout_prevention_rate_pct' in df.columns
    assert 'expected_blackout_loss_usd' in df.columns
    cgt_row = df[df['model_name'].str.contains('CGT-v3')].iloc[0]
    uncontrolled_row = df[df['model_name'].str.contains('Uncontrolled')].iloc[0]
    assert cgt_row['blackout_prevention_rate_pct'] > uncontrolled_row['blackout_prevention_rate_pct']
    assert cgt_row['expected_blackout_loss_usd'] < uncontrolled_row['expected_blackout_loss_usd']
