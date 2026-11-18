"""
gradual_drift_sim.py - Gradual Concept Drift Simulation Engine for CGT
======================================================================
Simulates continuous prequential behavior of:
  - Always-HT (Single tree baseline)
  - Always-ARF (10-tree ensemble baseline)
  - CGT-v3 (Learned Meta-Gate with ADWIN Cooldown & Budget Throttling)

Under 3 realistic gradual drift regimes over a 720-hour (30-day) observation window:
  1. Gradual Heatwave Demand Surge (Thermal Ramp)
  2. Gradual Renewable Decoupling (Wind/Solar Phase-down)
  3. Structural Industrial Shift (Load Profile Inversion)
"""

import numpy as np
import pandas as pd


def generate_gradual_drift_scenario(scenario_name="Gradual Heatwave Demand Surge", total_hours=720, seed=42):
    """
    Generates realistic 720-hour gradual concept drift dataset with dynamic features,
    ground truth labels, and simulated internal states for HT, ARF, and CGT-v3.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(total_hours)

    # 1. Base Grid Signal (Hourly with diurnal 24h cycle and weekly 168h rhythm)
    base_diurnal = 4200.0 * np.sin(2 * np.pi * t / 24.0 - np.pi / 2)  # peak at 18:00
    base_weekly = 1800.0 * np.sin(2 * np.pi * t / 168.0)
    noise = rng.normal(0, 450.0, size=total_hours)

    # 2. Transition parameter alpha_t (Sigmoid transition centered at t=300 with width=45 hours)
    if "Heatwave" in scenario_name:
        center_t = 300
        scale_t = 45.0
        # Drift: Base load shifts upward by +32%, diurnal amplitude expands by +25%
        alpha = 1.0 / (1.0 + np.exp(-(t - center_t) / scale_t))
        demand = 28000.0 + base_diurnal + base_weekly + noise + (alpha * 9000.0) + (alpha * base_diurnal * 0.25)
        # Target label: 1 if positive ramp (demand exceeds rolling median)
        threshold = 28000.0 + (alpha * 9000.0)
        prob_y1 = 1.0 / (1.0 + np.exp(-(demand - threshold) / 1200.0))
        y_true = (rng.rand(total_hours) < prob_y1).astype(int)

    elif "Renewable" in scenario_name:
        center_t = 280
        scale_t = 40.0
        alpha = 1.0 / (1.0 + np.exp(-(t - center_t) / scale_t))
        # Solar & Wind drop from 35% of total generation down to 5%
        renewables = (1.0 - alpha * 0.85) * (6500.0 + 3000.0 * np.sin(2 * np.pi * t / 24.0)) + rng.normal(0, 300, total_hours)
        demand = 29500.0 + base_diurnal + noise
        prob_y1 = 0.5 + 0.35 * np.sin(2 * np.pi * t / 24.0) + (alpha * 0.20 * (rng.rand(total_hours) - 0.5))
        prob_y1 = np.clip(prob_y1, 0.05, 0.95)
        y_true = (rng.rand(total_hours) < prob_y1).astype(int)

    else:  # Structural Industrial Shift
        center_t = 320
        scale_t = 50.0
        alpha = 1.0 / (1.0 + np.exp(-(t - center_t) / scale_t))
        # Night shift base load increases by +40%
        night_boost = 3500.0 * (1.0 - np.sin(2 * np.pi * t / 24.0))
        demand = 27000.0 + base_diurnal + noise + (alpha * night_boost)
        prob_y1 = 1.0 / (1.0 + np.exp(-(demand - 29000.0) / 1500.0))
        y_true = (rng.rand(total_hours) < prob_y1).astype(int)

    # 3. Simulate Policy Internal States
    # (a) Single Hoeffding Tree (HT) Baseline
    # HT margin: high confidence initially (~0.44), degrades to ~0.08 during drift peak,
    # then adapts slowly via leaf splitting over 200+ hours
    ht_adaptation = np.zeros(total_hours)
    for i in range(total_hours):
        if i > center_t:
            # Slow asymptotic tree leaf splitting adaptation
            ht_adaptation[i] = 1.0 - np.exp(-(i - center_t) / 160.0)

    ht_margin = 0.44 * (1.0 - alpha * 0.75) + (ht_adaptation * 0.28) + rng.normal(0, 0.025, total_hours)
    ht_margin = np.clip(ht_margin, 0.04, 0.52)

    # HT accuracy: drops from ~87% down to ~62% during drift, slowly recovers to ~85%
    ht_acc_prob = 0.87 - (alpha * 0.25 * (1.0 - ht_adaptation))
    y_pred_ht = np.where(rng.rand(total_hours) < ht_acc_prob, y_true, 1 - y_true)

    # (b) Always-ARF Ensemble (10 trees, continuous heavy training)
    # Stays consistently resilient (~89-91%), but requires 4,040 us continuously
    arf_acc_prob = 0.905 - (alpha * 0.03 * (1.0 - ht_adaptation * 0.8))
    y_pred_arf = np.where(rng.rand(total_hours) < arf_acc_prob, y_true, 1 - y_true)

    # (c) CGT-v3 Learned Meta-Gate with ADWIN & Budget Governance
    # Meta-score z_hat: estimates P(ARF beats HT) from margin, sliding error, and entropy
    # When margin is high (~0.44 in Phase 1), z_hat is low (~0.25 < tau=0.50).
    # When margin collapses under drift to ~0.08, z_hat escalates past 0.70.
    z_hat = 0.85 - (ht_margin * 1.35) + rng.normal(0, 0.035, total_hours)
    z_hat = np.clip(z_hat, 0.08, 0.92)

    # ADWIN Change Detector Simulation on HT prediction errors
    adwin_width = 32
    sliding_err = np.zeros(total_hours)
    adwin_alarm = np.zeros(total_hours, dtype=int)
    gate_state = np.zeros(total_hours, dtype=int)
    cooldown_counter = 0

    alarm_fired_at = None
    adwin_alarm_threshold = 0.23  # Error rate jump threshold

    for i in range(total_hours):
        # Sliding error of HT over past 40 steps
        w_start = max(0, i - 40)
        sliding_err[i] = (y_pred_ht[w_start:i+1] != y_true[w_start:i+1]).mean() if i > 0 else 0.13

        # ADWIN alarm trigger condition
        if alarm_fired_at is None and i > center_t - 20 and sliding_err[i] > adwin_alarm_threshold:
            adwin_alarm[i] = 1
            alarm_fired_at = i
            cooldown_counter = 50  # 50-step cooldown awakening

        # Gate Actuation Logic
        # Condition 1: ADWIN cooldown active -> AWAKE (ARF)
        if cooldown_counter > 0:
            gate_state[i] = 1
            cooldown_counter -= 1
        # Condition 2: Uncertainty-triggered (z_hat > 0.50) subject to duty cycle budget rho <= 15%
        else:
            recent_duty = gate_state[max(0, i - 100):i].mean() if i > 0 else 0.0
            if z_hat[i] > 0.50 and recent_duty < 0.15:
                gate_state[i] = 1
            else:
                gate_state[i] = 0

    # Final CGT-v3 predictions: if gate=1 use ARF, else use HT
    y_pred_cgt = np.where(gate_state == 1, y_pred_arf, y_pred_ht)

    # Compute rolling prequential accuracy (window = 48 hours for high responsiveness)
    roll_w = 48
    roll_acc_ht = pd.Series(y_pred_ht == y_true).rolling(roll_w, min_periods=10).mean().values * 100.0
    roll_acc_arf = pd.Series(y_pred_arf == y_true).rolling(roll_w, min_periods=10).mean().values * 100.0
    roll_acc_cgt = pd.Series(y_pred_cgt == y_true).rolling(roll_w, min_periods=10).mean().values * 100.0

    # Latency per step (us)
    # HT: ~110 us, ARF: ~4,040 us, CGT: 110 us when dormant + 4,040 us when awake
    lat_ht = np.full(total_hours, 110.0) + rng.normal(0, 8.0, total_hours)
    lat_arf = np.full(total_hours, 4040.0) + rng.normal(0, 120.0, total_hours)
    lat_cgt = np.where(gate_state == 1, 4040.0 + 80.0, 110.0 + 35.0) + rng.normal(0, 12.0, total_hours)

    # Duty cycle cumulative
    cum_duty = np.cumsum(gate_state) / (t + 1.0) * 100.0

    # Operational Phase annotation for each step
    phases = []
    for i in range(total_hours):
        if i < center_t - 80:
            phases.append("Phase 1: Calm Baseline (Dormant)")
        elif i < (alarm_fired_at if alarm_fired_at is not None else center_t + 30):
            phases.append("Phase 2: Incipient Drift & Margin Decay")
        elif i < (alarm_fired_at + 55 if alarm_fired_at is not None else center_t + 85):
            phases.append("Phase 3: ADWIN Alarm & Cooldown Awakening")
        else:
            phases.append("Phase 4: Post-Drift Convergence (Dormancy Restored)")

    df = pd.DataFrame({
        "step": t,
        "hour_label": [f"Hour {h}" for h in t],
        "demand_mw": demand,
        "alpha_drift": alpha,
        "y_true": y_true,
        "y_pred_ht": y_pred_ht,
        "y_pred_arf": y_pred_arf,
        "y_pred_cgt": y_pred_cgt,
        "ht_margin": ht_margin,
        "z_hat": z_hat,
        "sliding_error": sliding_err,
        "adwin_alarm": adwin_alarm,
        "gate_state": gate_state,
        "roll_acc_ht": roll_acc_ht,
        "roll_acc_arf": roll_acc_arf,
        "roll_acc_cgt": roll_acc_cgt,
        "lat_ht": lat_ht,
        "lat_arf": lat_arf,
        "lat_cgt": lat_cgt,
        "cum_duty_pct": cum_duty,
        "phase": phases
    })

    return df, alarm_fired_at, center_t

