"""
Unit tests for the Gradual Concept Drift simulation engine and CGT actuation dynamics.
"""

import pytest
import numpy as np
from src.analysis.gradual_drift_sim import generate_gradual_drift_scenario


def test_gradual_drift_data_structure():
    """Verify that gradual drift simulation generates a complete 720-hour (30-day) dataset."""
    df, alarm_t, center_t = generate_gradual_drift_scenario("Gradual Heatwave Demand Surge (+32% Ramp)")
    assert len(df) == 720
    assert alarm_t is not None
    assert 200 < alarm_t < 450
    assert center_t == 300

    # Required tracking columns must exist
    required_cols = [
        "step", "demand_mw", "alpha_drift", "ht_margin", "z_hat",
        "gate_state", "adwin_alarm", "roll_acc_ht", "roll_acc_arf",
        "roll_acc_cgt", "cum_duty_pct", "phase"
    ]
    for col in required_cols:
        assert col in df.columns


def test_cgt_actuation_and_latency_savings_under_gradual_drift():
    """Verify that CGT holds duty cycle and saves significant compute vs Always-ARF under gradual drift."""
    df, alarm_t, center_t = generate_gradual_drift_scenario("Gradual Heatwave Demand Surge (+32% Ramp)")

    # Overall duty cycle must remain controlled (under 25% even with 50-step cooldown)
    overall_duty = df["gate_state"].mean() * 100.0
    assert overall_duty < 25.0

    # CGT mean CPU latency must be at least 65% lower than Always-ARF
    mean_lat_cgt = df["lat_cgt"].mean()
    mean_lat_arf = df["lat_arf"].mean()
    assert mean_lat_cgt < 0.35 * mean_lat_arf

    # Phase 1: Pre-drift gate duty must be near zero
    p1_mask = df["step"] < (center_t - 80)
    p1_duty = df.loc[p1_mask, "gate_state"].mean() * 100.0
    assert p1_duty < 5.0

    # Phase 3: Cooldown awakening must have 100% active ensemble
    cooldown_mask = (df["step"] >= alarm_t) & (df["step"] < alarm_t + 50)
    assert df.loc[cooldown_mask, "gate_state"].all()

