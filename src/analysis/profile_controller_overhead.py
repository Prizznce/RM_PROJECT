"""
Task B: Micro-Component Profiling of CGT-v1, CGT-v2, and CGT-v3.
Isolates per-component latencies using time.perf_counter_ns():
  1. HT predict + learn
  2. Feature extraction (margins, ratios, cyclical calendar)
  3. Meta-model predict + update (StandardScaler + LogisticRegression)
  4. ADWIN update
  5. ARF predict (deployed inference vs shadow training inference)
  6. ARF learn & buffer replay
  7. Bookkeeping (deques, counters, budget checks)
"""

import time
import yaml
import numpy as np
import pandas as pd
from collections import deque
from river.drift import ADWIN
from river.linear_model import LogisticRegression
from river.preprocessing import StandardScaler

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFOptimizedWrapper

def profile_controller_components(controller_type="v3", n_instances=5000):
    with open("configs/frozen_v3.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    loader = ElectricityStreamLoader(csv_path="data/energy_dataset_clean.csv")
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_len = cfg.get("data", {}).get("val_instances", 10512)
    stream = full_stream[val_len:val_len + n_instances]

    # Components
    ht = HoeffdingTreeWrapper()
    arf = ARFOptimizedWrapper(seed=42)
    detector = ADWIN(delta=0.005, clock=32)
    meta_gate = StandardScaler() | LogisticRegression()
    buffer = deque(maxlen=200)
    recent_errors = deque(maxlen=50)

    # Accumulators (nanoseconds)
    time_ht = 0
    time_feat = 0
    time_meta = 0
    time_adwin = 0
    time_arf_pred = 0
    time_arf_learn = 0
    time_bookkeeping = 0

    cooldown = 0
    gate_open_steps = 0
    tau_gate = 0.20
    max_rho = 0.10
    rng = np.random.RandomState(42)

    for step, (x, y, ts) in enumerate(stream, start=1):
        # 1. Bookkeeping & budget check
        t0 = time.perf_counter_ns()
        current_rho = gate_open_steps / max(step, 1)
        budget_ok = current_rho < max_rho
        time_bookkeeping += time.perf_counter_ns() - t0

        # 2. HT predict & margin extraction
        t0 = time.perf_counter_ns()
        proba = ht.model.predict_proba_one(x)
        p0 = proba.get(0, 0.5)
        p1 = proba.get(1, 0.5)
        ht_margin = abs(p1 - p0)
        pred_ht = 1 if p1 > p0 else 0
        time_ht += time.perf_counter_ns() - t0

        if controller_type in ["v2", "v3"]:
            # Feature extraction
            t0 = time.perf_counter_ns()
            err_rate = sum(recent_errors) / len(recent_errors) if recent_errors else 0.15
            meta_feat = {
                'margin': float(ht_margin),
                'recent_err': float(err_rate),
                'hour_sin': float(x.get('hour_sin', 0.0)),
                'hour_cos': float(x.get('hour_cos', 0.0)),
                'ren_ratio': float(x.get('renewable_ratio_lag_1', 0.2)),
                'd_diff_24': float(x.get('demand_diff_1_24', 0.0)) / 5000.0,
                'in_cooldown': 1.0 if cooldown > 0 else 0.0
            }
            time_feat += time.perf_counter_ns() - t0

        # Gate trigger decision
        gate_open = False
        if controller_type == "v1":
            gate_open = (cooldown > 0)
        elif controller_type == "v2":
            gate_open = (cooldown > 0) or (ht_margin < 0.25 and budget_ok)
        elif controller_type == "v3":
            t0 = time.perf_counter_ns()
            prob_meta = meta_gate.predict_proba_one(meta_feat).get(1, 0.0)
            learned_trigger = prob_meta > tau_gate
            time_meta += time.perf_counter_ns() - t0
            gate_open = (cooldown > 0) or (learned_trigger and budget_ok)

        if gate_open:
            gate_open_steps += 1
            t0 = time.perf_counter_ns()
            y_pred = arf.predict_one(x)
            time_arf_pred += time.perf_counter_ns() - t0
        else:
            y_pred = pred_ht

        # Learn step
        err = 1.0 if y_pred != y else 0.0

        # ADWIN detector update
        t0 = time.perf_counter_ns()
        detector.update(err)
        drift_detected = detector.drift_detected
        time_adwin += time.perf_counter_ns() - t0

        if drift_detected:
            cooldown = 50
            gate_open = True
            t0 = time.perf_counter_ns()
            for bx, by in buffer:
                arf.learn_one(bx, by)
            time_arf_learn += time.perf_counter_ns() - t0

        if controller_type == "v3":
            # Shadow ARF inference for meta-target z_t
            t0 = time.perf_counter_ns()
            pred_arf_shadow = arf.predict_one(x)
            time_arf_pred += time.perf_counter_ns() - t0

            # Meta-model learning update
            t0 = time.perf_counter_ns()
            err_ht = 1.0 if pred_ht != y else 0.0
            err_arf = 1.0 if pred_arf_shadow != y else 0.0
            z_t = 1 if (err_ht == 1.0 and err_arf == 0.0) else 0
            meta_gate.learn_one(meta_feat, z_t)
            time_meta += time.perf_counter_ns() - t0

        # Base learner updates
        t0 = time.perf_counter_ns()
        ht.learn_one(x, y)
        time_ht += time.perf_counter_ns() - t0

        t0 = time.perf_counter_ns()
        if gate_open:
            arf.learn_one(x, y)
        else:
            buffer.append((x, y))
            if controller_type == "v3" and rng.rand() < 0.05:
                arf.learn_one(x, y)
        time_arf_learn += time.perf_counter_ns() - t0

        t0 = time.perf_counter_ns()
        recent_errors.append(1.0 if pred_ht != y else 0.0)
        if cooldown > 0:
            cooldown -= 1
        time_bookkeeping += time.perf_counter_ns() - t0

    N = len(stream)
    return {
        "controller": f"CGT-{controller_type}",
        "ht_predict_and_learn_us": round(time_ht / N / 1000.0, 2),
        "feature_extraction_us": round(time_feat / N / 1000.0, 2),
        "meta_model_predict_update_us": round(time_meta / N / 1000.0, 2),
        "adwin_update_us": round(time_adwin / N / 1000.0, 2),
        "arf_predict_us": round(time_arf_pred / N / 1000.0, 2),
        "arf_learn_and_replay_us": round(time_arf_learn / N / 1000.0, 2),
        "bookkeeping_us": round(time_bookkeeping / N / 1000.0, 2),
        "total_profiled_us": round((time_ht + time_feat + time_meta + time_adwin + time_arf_pred + time_arf_learn + time_bookkeeping) / N / 1000.0, 2)
    }

def run_profiling():
    print("=================================================================", flush=True)
    print("TASK B: DETAILED LATENCY COMPONENT PROFILING (CGT-v1 vs v2 vs v3)", flush=True)
    print("=================================================================\n", flush=True)

    results = []
    for c_type in ["v1", "v2", "v3"]:
        print(f"Profiling CGT-{c_type} over 5,000 streaming instances...", flush=True)
        res = profile_controller_components(c_type, n_instances=5000)
        results.append(res)

    df = pd.DataFrame(results)
    print("\n--- Component Latency Breakdown (Mean us/instance) ---")
    print(df.to_string(index=False))

    df.to_csv("results/tables/day7_component_overhead_profiling.csv", index=False)
    print("\nSaved component breakdown to results/tables/day7_component_overhead_profiling.csv", flush=True)

if __name__ == "__main__":
    run_profiling()
