"""
Task B: Testing Proposed CGT-v3 Optimizations STRICTLY on the Validation Stream (10,512 instances).
Compares:
  - Baseline CGT-v3 (frozen parameters)
  - Opt A: HT-Error-Gated Shadow Evaluation (only query ARF for z_t when HT is wrong, since z_t=0 trivially when HT is right)
  - Opt B: Subsampled Shadow Evaluation (query ARF for z_t only on 10% of instances)
  - Opt C: Combined (HT-Error-Gated + 2% background probing)
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

class CGTv3OptimizedCandidate:
    def __init__(self, mode="opt_c", tau_gate=0.20, max_rho=0.10, h_cooldown=50, seed=42):
        self.mode = mode
        self.tau_gate = tau_gate
        self.max_rho = max_rho
        self.h_cooldown = h_cooldown
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.ht = HoeffdingTreeWrapper()
        self.arf = ARFOptimizedWrapper(seed=seed)
        self.detector = ADWIN(delta=0.005, clock=32)
        self.meta_gate = StandardScaler() | LogisticRegression()

        self.buffer = deque(maxlen=200)
        self.recent_ht_errors = deque(maxlen=50)
        self.cooldown_counter = 0
        self.gate_open = False

        self.total_steps = 0
        self.gate_open_steps = 0
        self.total_alarms = 0
        self._current_meta_features = None

    def _extract_meta_features(self, x):
        proba = self.ht.model.predict_proba_one(x)
        p0 = proba.get(0, 0.5)
        p1 = proba.get(1, 0.5)
        margin = abs(p1 - p0)
        err_rate = sum(self.recent_ht_errors) / len(self.recent_ht_errors) if self.recent_ht_errors else 0.15
        return {
            'margin': float(margin),
            'recent_err': float(err_rate),
            'hour_sin': float(x.get('hour_sin', 0.0)),
            'hour_cos': float(x.get('hour_cos', 0.0)),
            'ren_ratio': float(x.get('renewable_ratio_lag_1', 0.2)),
            'd_diff_24': float(x.get('demand_diff_1_24', 0.0)) / 5000.0,
            'in_cooldown': 1.0 if self.cooldown_counter > 0 else 0.0
        }

    def predict_one(self, x):
        self.total_steps += 1
        current_rho = self.gate_open_steps / max(self.total_steps, 1)
        budget_ok = current_rho < self.max_rho

        meta_feat = self._extract_meta_features(x)
        self._current_meta_features = meta_feat

        prob_meta = self.meta_gate.predict_proba_one(meta_feat).get(1, 0.0)
        learned_trigger = prob_meta > self.tau_gate

        if self.cooldown_counter > 0:
            self.gate_open = True
            self.cooldown_counter -= 1
        elif learned_trigger and budget_ok:
            self.gate_open = True
        else:
            self.gate_open = False

        if self.gate_open:
            self.gate_open_steps += 1
            return self.arf.predict_one(x)
        return self.ht.predict_one(x)

    def learn_one(self, x, y, y_pred):
        err = 1.0 if y_pred != y else 0.0
        self.detector.update(err)
        if self.detector.drift_detected:
            self.total_alarms += 1
            self.cooldown_counter = self.h_cooldown
            self.gate_open = True
            for bx, by in self.buffer:
                self.arf.learn_one(bx, by)

        pred_ht = self.ht.predict_one(x)
        err_ht = 1.0 if pred_ht != y else 0.0
        self.recent_ht_errors.append(err_ht)

        # Meta-gate training logic based on mode
        if self.mode == "baseline":
            # Original: evaluates ARF every step
            pred_arf = self.arf.predict_one(x)
            err_arf = 1.0 if pred_arf != y else 0.0
            z_t = 1 if (err_ht == 1.0 and err_arf == 0.0) else 0
            if self._current_meta_features is not None:
                self.meta_gate.learn_one(self._current_meta_features, z_t)
        elif self.mode == "opt_a":
            # Opt A: Error-Gated. If HT is correct, z_t is mathematically 0.
            if err_ht == 0.0:
                z_t = 0
            else:
                pred_arf = self.arf.predict_one(x)
                err_arf = 1.0 if pred_arf != y else 0.0
                z_t = 1 if err_arf == 0.0 else 0
            if self._current_meta_features is not None:
                self.meta_gate.learn_one(self._current_meta_features, z_t)
        elif self.mode == "opt_b":
            # Opt B: Subsampled at 20%
            if self.rng.rand() < 0.20 or self.gate_open:
                pred_arf = self.arf.predict_one(x) if not self.gate_open else y_pred
                err_arf = 1.0 if pred_arf != y else 0.0
                z_t = 1 if (err_ht == 1.0 and err_arf == 0.0) else 0
                if self._current_meta_features is not None:
                    self.meta_gate.learn_one(self._current_meta_features, z_t)
        elif self.mode == "opt_c":
            # Opt C: Error-Gated + reused y_pred if gate was open + 2% background probe
            if self.gate_open:
                err_arf = 1.0 if y_pred != y else 0.0
                z_t = 1 if (err_ht == 1.0 and err_arf == 0.0) else 0
            elif err_ht == 0.0:
                z_t = 0
            else:
                pred_arf = self.arf.predict_one(x)
                err_arf = 1.0 if pred_arf != y else 0.0
                z_t = 1 if err_arf == 0.0 else 0
            if self._current_meta_features is not None:
                self.meta_gate.learn_one(self._current_meta_features, z_t)

        self.ht.learn_one(x, y)
        if self.gate_open:
            self.arf.learn_one(x, y)
        else:
            self.buffer.append((x, y))
            probe_prob = 0.02 if "opt" in self.mode else 0.05
            if self.rng.rand() < probe_prob:
                self.arf.learn_one(x, y)

def evaluate_on_validation():
    print("=================================================================", flush=True)
    print("TASK B: EVALUATION OF PROPOSED OPTIMIZATIONS (STRICTLY ON VALIDATION)", flush=True)
    print("=================================================================\n", flush=True)

    with open("configs/frozen_v3.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_len = cfg.get("data", {}).get("val_instances", 10512)
    val_stream = full_stream[:val_len]

    modes = [
        ("baseline", "CGT-v3 Baseline (Frozen)"),
        ("opt_a", "CGT-v3 + HT-Error-Gated Shadow Query"),
        ("opt_b", "CGT-v3 + 20% Subsampled Shadow Query"),
        ("opt_c", "CGT-v3 + Error-Gated + Reuse Awake + 2% Probe")
    ]

    results = []
    for mode_id, mode_label in modes:
        controller = CGTv3OptimizedCandidate(mode=mode_id, seed=42)
        correct = 0
        t0 = time.perf_counter_ns()

        for x, y, ts in val_stream:
            y_pred = controller.predict_one(x)
            controller.learn_one(x, y, y_pred)
            if y_pred == y:
                correct += 1

        t_elapsed_ns = time.perf_counter_ns() - t0
        acc = (correct / val_len) * 100.0
        mean_us = (t_elapsed_ns / val_len) / 1000.0
        rho = controller.gate_open_steps / val_len

        print(f"[{mode_label}]: Acc = {acc:.2f}%, rho = {rho:.4f}, Mean CPU = {mean_us:.2f} us/inst", flush=True)
        results.append({
            "mode": mode_id,
            "label": mode_label,
            "validation_accuracy_pct": round(acc, 2),
            "gate_rho": round(rho, 4),
            "mean_cpu_us": round(mean_us, 2)
        })

    df = pd.DataFrame(results)
    df.to_csv("results/tables/day7_v3_optimization_validation_comparison.csv", index=False)
    print("\nSaved comparison to results/tables/day7_v3_optimization_validation_comparison.csv", flush=True)

if __name__ == "__main__":
    evaluate_on_validation()

