"""
Profile CGT per-step execution components.
Measures nanosecond execution times for:
1. HT Predict
2. HT Learn
3. Gate Feature Extraction
4. Gate Scoring (River LogisticRegression forward)
5. SGD Update (River SGD backward)
6. ADWIN Drift Detector Update
7. ARF Wakeup / Switch Logic
8. Replay Buffer Execution on Drift

Saves results to results/tables/cgt_v4_profile.csv.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from river.drift import ADWIN
from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier
from river.linear_model import LogisticRegression
from river.preprocessing import StandardScaler
from src.data.stream_loader import ElectricityStreamLoader

def run_cgt_profiling(n_steps=5000):
    print(f"Loading stream data for profiling ({n_steps} steps)...")
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    stream = list(loader.iter_stream(warmup_rows=24))[:n_steps]
    
    # Initialize components
    ht = HoeffdingTreeClassifier(grace_period=50)
    arf = ARFClassifier(n_models=3, grace_period=50, lambda_value=6, delta=0.01, seed=42)
    meta_gate = StandardScaler() | LogisticRegression()
    adwin = ADWIN(delta=0.005, clock=32)
    buffer = deque(maxlen=200)
    recent_errors = deque(maxlen=50)
    cooldown = 0
    
    # Timing arrays (in nanoseconds)
    t_ht_predict = np.zeros(n_steps, dtype=np.int64)
    t_ht_learn = np.zeros(n_steps, dtype=np.int64)
    t_gate_features = np.zeros(n_steps, dtype=np.int64)
    t_gate_scoring = np.zeros(n_steps, dtype=np.int64)
    t_sgd_update = np.zeros(n_steps, dtype=np.int64)
    t_adwin = np.zeros(n_steps, dtype=np.int64)
    t_arf_wake = np.zeros(n_steps, dtype=np.int64)
    t_replay = np.zeros(n_steps, dtype=np.int64)
    t_total_step = np.zeros(n_steps, dtype=np.int64)
    
    drift_triggers = 0
    gate_triggers = 0
    
    print("Executing component profiling loop...")
    for idx, (x_t, y_t, ts) in enumerate(stream):
        t_start_step = time.perf_counter_ns()
        
        # 1. HT Predict
        t0 = time.perf_counter_ns()
        proba_ht = ht.predict_proba_one(x_t)
        p0 = proba_ht.get(0, 0.5)
        p1 = proba_ht.get(1, 0.5)
        pred_ht = 1 if p1 > p0 else 0
        t_ht_predict[idx] = time.perf_counter_ns() - t0
        
        # 2. Gate Feature Extraction
        t0 = time.perf_counter_ns()
        margin = abs(p1 - p0)
        err_rate = sum(recent_errors) / len(recent_errors) if recent_errors else 0.15
        h_sin = x_t.get('hour_sin', 0.0)
        h_cos = x_t.get('hour_cos', 0.0)
        ren_ratio = x_t.get('renewable_ratio_lag_1', 0.2)
        d_diff_24 = x_t.get('demand_diff_1_24', 0.0) / 5000.0
        meta_feat = {
            'margin': float(margin),
            'recent_err': float(err_rate),
            'hour_sin': float(h_sin),
            'hour_cos': float(h_cos),
            'ren_ratio': float(ren_ratio),
            'd_diff_24': float(d_diff_24),
            'in_cooldown': 1.0 if cooldown > 0 else 0.0
        }
        t_gate_features[idx] = time.perf_counter_ns() - t0
        
        # 3. Gate Scoring (River Logistic Regression)
        t0 = time.perf_counter_ns()
        proba_meta = meta_gate.predict_proba_one(meta_feat)
        prob_arf_wins = proba_meta.get(1, 0.0)
        gate_decision = (prob_arf_wins > 0.25) or (cooldown > 0)
        if gate_decision:
            gate_triggers += 1
        t_gate_scoring[idx] = time.perf_counter_ns() - t0
        
        # 4. ARF Wakeup / Prediction
        t0 = time.perf_counter_ns()
        if gate_decision:
            pred = arf.predict_one(x_t)
            if pred is None:
                pred = pred_ht
        else:
            pred = pred_ht
        t_arf_wake[idx] = time.perf_counter_ns() - t0
        
        # 5. ADWIN concept drift update
        t0 = time.perf_counter_ns()
        err = 1.0 if pred != y_t else 0.0
        adwin.update(err)
        drift_now = adwin.drift_detected
        if drift_now:
            drift_triggers += 1
            cooldown = 50
        elif cooldown > 0:
            cooldown -= 1
        t_adwin[idx] = time.perf_counter_ns() - t0
        
        # 6. Replay Buffer on drift
        t0 = time.perf_counter_ns()
        if drift_now:
            for bx, by in buffer:
                arf.learn_one(bx, by)
        t_replay[idx] = time.perf_counter_ns() - t0
        
        # 7. Meta-gate SGD Update
        t0 = time.perf_counter_ns()
        err_ht = 1.0 if pred_ht != y_t else 0.0
        recent_errors.append(err_ht)
        pred_arf = arf.predict_one(x_t)
        err_arf = 1.0 if pred_arf != y_t else 0.0
        z_t = 1 if (err_ht == 1.0 and err_arf == 0.0) else 0
        meta_gate.learn_one(meta_feat, z_t)
        t_sgd_update[idx] = time.perf_counter_ns() - t0
        
        # 8. HT Learn & Buffer Management
        t0 = time.perf_counter_ns()
        ht.learn_one(x_t, y_t)
        if gate_decision:
            arf.learn_one(x_t, y_t)
        else:
            buffer.append((x_t, y_t))
        t_ht_learn[idx] = time.perf_counter_ns() - t0
        
        t_total_step[idx] = time.perf_counter_ns() - t_start_step
        
    print(f"Profiling complete. Total steps: {n_steps}, Gate triggers: {gate_triggers}, Drift alerts: {drift_triggers}")
    
    # Compile summary table
    components = [
        ("HT Predict", t_ht_predict),
        ("HT Learn", t_ht_learn),
        ("Gate Feature Extraction", t_gate_features),
        ("Gate Scoring (River Logistic)", t_gate_scoring),
        ("SGD Update (River Meta-Gate)", t_sgd_update),
        ("ADWIN Drift Detector", t_adwin),
        ("ARF Wakeup / Scoring", t_arf_wake),
        ("Replay Buffer (Blocking on Drift)", t_replay),
        ("TOTAL STEP", t_total_step)
    ]
    
    records = []
    total_mean_ns = np.mean(t_total_step)
    
    for name, arr in components:
        mean_us = np.mean(arr) / 1000.0
        p50_us = np.percentile(arr, 50) / 1000.0
        p95_us = np.percentile(arr, 95) / 1000.0
        p99_us = np.percentile(arr, 99) / 1000.0
        max_us = np.max(arr) / 1000.0
        pct_share = (np.mean(arr) / total_mean_ns) * 100.0 if name != "TOTAL STEP" else 100.0
        
        records.append({
            "Component": name,
            "Mean_us": round(mean_us, 2),
            "p50_us": round(p50_us, 2),
            "p95_us": round(p95_us, 2),
            "p99_us": round(p99_us, 2),
            "Max_us": round(max_us, 2),
            "Share_of_Total_Pct": round(pct_share, 1)
        })
        
    df_prof = pd.DataFrame(records)
    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cgt_v4_profile.csv")
    df_prof.to_csv(out_path, index=False)
    print("\n" + "=" * 80)
    print("CGT PER-STEP COMPONENT PROFILE (results/tables/cgt_v4_profile.csv):")
    print("=" * 80)
    print(df_prof.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    run_cgt_profiling(n_steps=5000)

