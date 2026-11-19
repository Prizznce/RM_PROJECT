import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import math
import hashlib
import tempfile
import numpy as np
import pandas as pd
from collections import deque

# Matplotlib configuration for sandbox/temp cache
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import matplotlib.pyplot as plt

# River imports
from river.drift import ADWIN
from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier
from river.linear_model import LogisticRegression
from river.preprocessing import StandardScaler

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper, ARFOptimizedWrapper
from src.controllers.cgt_v3 import CGTv3LearnedGateController


class ARF3Wrapper:
    """Standardized ARF with 3 trees for fair benchmarking."""
    def __init__(self, grace_period=50, lambda_value=6, delta=0.01, split_criterion='info_gain', seed=42):
        self.grace_period = grace_period
        self.lambda_value = lambda_value
        self.delta = delta
        self.split_criterion = split_criterion
        self.seed = seed
        self.name = "ARF-3 (3 Trees)"
        self.model = ARFClassifier(
            n_models=3,
            grace_period=grace_period,
            lambda_value=lambda_value,
            delta=delta,
            split_criterion=split_criterion,
            seed=seed
        )

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y, y_pred=None):
        self.model.learn_one(x, y)


class RandomGateBaseline:
    """
    Random-Gate Baseline:
    Routes to ARF-3 with fixed probability p at each time step (Bernoulli trial),
    otherwise routes to Hoeffding Tree.
    """
    def __init__(self, p=0.07, arf_config=None, seed=42):
        self.p = p
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.name = f"Random-Gate (p={p:.2%})"
        
        self.ht = HoeffdingTreeWrapper()
        arf_cfg = arf_config or {"grace_period": 50, "lambda_value": 6, "delta": 0.01}
        self.arf = ARF3Wrapper(
            grace_period=arf_cfg.get("grace_period", 50),
            lambda_value=arf_cfg.get("lambda_value", 6),
            delta=arf_cfg.get("delta", 0.01),
            seed=seed
        )
        
        self.total_steps = 0
        self.gate_open_steps = 0
        self.gate_open = False

    def predict_one(self, x):
        self.total_steps += 1
        self.gate_open = bool(self.rng.rand() < self.p)
        if self.gate_open:
            self.gate_open_steps += 1
            return self.arf.predict_one(x)
        return self.ht.predict_one(x)

    def learn_one(self, x, y, y_pred=None):
        self.ht.learn_one(x, y)
        if self.gate_open:
            self.arf.learn_one(x, y)


def evaluate_stream_prequential(model, stream, is_timed=False):
    """
    Standardized prequential evaluation: predict_one -> learn_one.
    If is_timed=True, captures controlled per-instance nanosecond timing.
    """
    n = len(stream)
    y_trues = np.zeros(n, dtype=int)
    y_preds = np.zeros(n, dtype=int)
    gate_states = np.zeros(n, dtype=int)
    cpu_ns = np.zeros(n, dtype=int) if is_timed else None
    
    correct = 0
    t_start_total = time.perf_counter_ns()
    
    for idx, (x_t, y_t, ts) in enumerate(stream):
        y_trues[idx] = y_t
        
        if is_timed:
            t0 = time.perf_counter_ns()
            pred = model.predict_one(x_t)
            if hasattr(model, 'name') and 'CGT' in model.name:
                model.learn_one(x_t, y_t, pred)
            else:
                model.learn_one(x_t, y_t)
            dt = time.perf_counter_ns() - t0
            cpu_ns[idx] = dt
        else:
            pred = model.predict_one(x_t)
            if hasattr(model, 'name') and 'CGT' in model.name:
                model.learn_one(x_t, y_t, pred)
            else:
                model.learn_one(x_t, y_t)
                
        y_preds[idx] = pred
        is_gate = 1 if getattr(model, 'gate_open', False) else (1 if isinstance(model, ARF3Wrapper) or isinstance(model, ARFFullWrapper) else 0)
        gate_states[idx] = is_gate
        
        if pred == y_t:
            correct += 1

    t_total_ns = time.perf_counter_ns() - t_start_total
    acc = (correct / n) * 100.0
    mean_cpu_us = (t_total_ns / n) / 1000.0
    duty_cycle = float(np.mean(gate_states))
    
    res = {
        "accuracy_pct": round(acc, 3),
        "mean_cpu_us": round(mean_cpu_us, 2),
        "duty_cycle": round(duty_cycle, 4),
        "correct_count": correct,
        "total_instances": n,
        "y_trues": y_trues,
        "y_preds": y_preds,
        "gate_states": gate_states
    }
    if is_timed:
        res["cpu_ns"] = cpu_ns
        res["latency_p50_us"] = round(float(np.percentile(cpu_ns, 50)) / 1000.0, 2)
        res["latency_p95_us"] = round(float(np.percentile(cpu_ns, 95)) / 1000.0, 2)
        res["latency_p99_us"] = round(float(np.percentile(cpu_ns, 99)) / 1000.0, 2)
        res["latency_max_us"] = round(float(np.max(cpu_ns)) / 1000.0, 2)
    return res


def main():
    print("=" * 80)
    print("FAIR COMPARISON BENCHMARK: ARF-3 vs CGT (IDENTICAL STREAMING PROTOCOL)")
    print("=" * 80)

    # 1. Load Data & Create Strict Splits
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    full_stream = list(loader.iter_stream(warmup_rows=24))
    n_full = len(full_stream)
    val_instances = 10512
    test_instances = 24528
    
    val_stream = full_stream[:val_instances]
    test_stream = full_stream[val_instances:val_instances + test_instances]
    
    print(f"Total Stream Instances: {n_full}")
    print(f"Validation Split:       {len(val_stream)} instances (Days 1 - 439)")
    print(f"Test Split:             {len(test_stream)} instances (Days 440 - 1461)")
    print("-" * 80)

    # =========================================================================
    # STEP 1: VALIDATION TUNING (K = 16 CANDIDATES EACH)
    # =========================================================================
    print("\n[STEP 1/7] Running Fair Validation Hyperparameter Tuning (K = 16 each)...")
    
    # ARF-3 Candidates (16 grid combinations)
    arf3_grid = []
    for gp in [25, 50, 100, 200]:
        for lv in [4, 6]:
            for d in [0.01, 0.001]:
                arf3_grid.append({
                    "grace_period": gp,
                    "lambda_value": lv,
                    "delta": d,
                    "split_criterion": "info_gain",
                    "seed": 42
                })
    print(f"  -> Generated {len(arf3_grid)} ARF-3 hyperparameter candidate configurations.")

    # CGT Candidates (16 grid combinations)
    cgt_grid = []
    for tau in [0.15, 0.20, 0.25, 0.30]:
        for max_rho in [0.08, 0.12]:
            for cd in [50, 100]:
                cgt_grid.append({
                    "tau_gate": tau,
                    "max_rho": max_rho,
                    "h_cooldown": cd,
                    "buffer_b": 200,
                    "adwin_delta": 0.005,
                    "seed": 42
                })
    print(f"  -> Generated {len(cgt_grid)} CGT hyperparameter candidate configurations.")

    # Evaluate ARF-3 candidates on validation stream
    val_tuning_records = []
    print("\n  Evaluating ARF-3 Candidates on Validation Stream (N = 10,512)...")
    best_arf3_config = None
    best_arf3_val_acc = -1.0
    
    for i, cfg in enumerate(arf3_grid):
        model = ARF3Wrapper(
            grace_period=cfg["grace_period"],
            lambda_value=cfg["lambda_value"],
            delta=cfg["delta"],
            split_criterion=cfg["split_criterion"],
            seed=cfg["seed"]
        )
        res = evaluate_stream_prequential(model, val_stream, is_timed=False)
        print(f"    ARF-3 Trial {i+1:02d}/16: gp={cfg['grace_period']}, lv={cfg['lambda_value']}, delta={cfg['delta']} -> Acc = {res['accuracy_pct']}%, CPU = {res['mean_cpu_us']} us/inst")
        
        record = {
            "model_family": "ARF-3",
            "trial_id": i + 1,
            "params": str(cfg),
            "val_accuracy_pct": res["accuracy_pct"],
            "val_cpu_us": res["mean_cpu_us"],
            "val_duty_cycle": res["duty_cycle"]
        }
        val_tuning_records.append(record)
        
        if res["accuracy_pct"] > best_arf3_val_acc:
            best_arf3_val_acc = res["accuracy_pct"]
            best_arf3_config = cfg

    print(f"  >>> BEST ARF-3 VALIDATION CONFIG: Acc = {best_arf3_val_acc}% -> {best_arf3_config}")

    # Evaluate CGT candidates on validation stream
    print("\n  Evaluating CGT Candidates on Validation Stream (N = 10,512)...")
    best_cgt_config = None
    best_cgt_val_acc = -1.0
    
    # We equip CGT's heavy arm with the tuned ARF-3 model architecture
    for i, cfg in enumerate(cgt_grid):
        arf_heavy = ARF3Wrapper(
            grace_period=best_arf3_config["grace_period"],
            lambda_value=best_arf3_config["lambda_value"],
            delta=best_arf3_config["delta"],
            seed=42
        )
        cgt_model = CGTv3LearnedGateController(
            tau_gate=cfg["tau_gate"],
            max_rho=cfg["max_rho"],
            h_cooldown=cfg["h_cooldown"],
            buffer_b=cfg["buffer_b"],
            adwin_delta=cfg["adwin_delta"],
            arf_model=arf_heavy,
            seed=cfg["seed"]
        )
        res = evaluate_stream_prequential(cgt_model, val_stream, is_timed=False)
        print(f"    CGT Trial {i+1:02d}/16: tau={cfg['tau_gate']}, rho_max={cfg['max_rho']}, cd={cfg['h_cooldown']} -> Acc = {res['accuracy_pct']}%, rho = {res['duty_cycle']:.3f}, CPU = {res['mean_cpu_us']} us/inst")
        
        record = {
            "model_family": "CGT",
            "trial_id": i + 1,
            "params": str(cfg),
            "val_accuracy_pct": res["accuracy_pct"],
            "val_cpu_us": res["mean_cpu_us"],
            "val_duty_cycle": res["duty_cycle"]
        }
        val_tuning_records.append(record)
        
        if res["accuracy_pct"] > best_cgt_val_acc:
            best_cgt_val_acc = res["accuracy_pct"]
            best_cgt_config = cfg

    print(f"  >>> BEST CGT VALIDATION CONFIG: Acc = {best_cgt_val_acc}% -> {best_cgt_config}")

    # Save validation tuning table
    os.makedirs("results/tables", exist_ok=True)
    df_val_tuning = pd.DataFrame(val_tuning_records)
    df_val_tuning.to_csv("results/tables/fair_comparison_val_tuning.csv", index=False)

    # Freeze configurations & compute SHA-256 hash BEFORE touching the test stream
    frozen_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_instances": len(val_stream),
        "test_instances": len(test_stream),
        "search_budget_K": 16,
        "best_arf3_frozen_config": best_arf3_config,
        "best_arf3_val_accuracy_pct": best_arf3_val_acc,
        "best_cgt_frozen_config": best_cgt_config,
        "best_cgt_val_accuracy_pct": best_cgt_val_acc
    }
    frozen_json_str = json.dumps(frozen_payload, indent=2, sort_keys=True)
    frozen_hash = hashlib.sha256(frozen_json_str.encode('utf-8')).hexdigest()
    frozen_payload["sha256_hash"] = frozen_hash

    os.makedirs("configs", exist_ok=True)
    with open("configs/fair_comparison_frozen_configs.json", "w", encoding="utf-8") as f:
        json.dump(frozen_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("FROZEN HYPERPARAMETERS LOCKED & HASHED:")
    print(f"  Config File: configs/fair_comparison_frozen_configs.json")
    print(f"  SHA-256:     {frozen_hash}")
    print("=" * 80 + "\n")

    # =========================================================================
    # STEP 2: TEST STREAM EVALUATION (SINGLE PASS, N = 24,528)
    # =========================================================================
    print("[STEP 2/7] Running Frozen Models on 24,528-Hour Test Stream (Controlled Timing)...")
    
    # 1. Run Frozen ARF-3
    print("  Executing Frozen ARF-3 on Test Stream...")
    frozen_arf3 = ARF3Wrapper(
        grace_period=best_arf3_config["grace_period"],
        lambda_value=best_arf3_config["lambda_value"],
        delta=best_arf3_config["delta"],
        split_criterion=best_arf3_config["split_criterion"],
        seed=best_arf3_config["seed"]
    )
    res_arf3 = evaluate_stream_prequential(frozen_arf3, test_stream, is_timed=True)
    print(f"  -> ARF-3 Test Acc:     {res_arf3['accuracy_pct']}%")
    print(f"  -> ARF-3 Duty Cycle:   {res_arf3['duty_cycle'] * 100.0:.2f}%")
    print(f"  -> ARF-3 Mean Latency: {res_arf3['mean_cpu_us']} us/inst (Worst: {res_arf3['latency_max_us']} us)")

    # 2. Run Frozen CGT
    print("\n  Executing Frozen CGT on Test Stream...")
    cgt_heavy = ARF3Wrapper(
        grace_period=best_arf3_config["grace_period"],
        lambda_value=best_arf3_config["lambda_value"],
        delta=best_arf3_config["delta"],
        seed=42
    )
    frozen_cgt = CGTv3LearnedGateController(
        tau_gate=best_cgt_config["tau_gate"],
        max_rho=best_cgt_config["max_rho"],
        h_cooldown=best_cgt_config["h_cooldown"],
        buffer_b=best_cgt_config["buffer_b"],
        adwin_delta=best_cgt_config["adwin_delta"],
        arf_model=cgt_heavy,
        seed=best_cgt_config["seed"]
    )
    res_cgt = evaluate_stream_prequential(frozen_cgt, test_stream, is_timed=True)
    rho_cgt_empirical = res_cgt["duty_cycle"]
    print(f"  -> CGT Test Acc:       {res_cgt['accuracy_pct']}%")
    print(f"  -> CGT Duty Cycle:     {res_cgt['duty_cycle'] * 100.0:.2f}% (Empirical rho = {rho_cgt_empirical:.4f})")
    print(f"  -> CGT Mean Latency:   {res_cgt['mean_cpu_us']} us/inst (Worst: {res_cgt['latency_max_us']} us)")

    # 3. Reference: Always-HT
    print("\n  Executing Reference Always-HT on Test Stream...")
    ref_ht = HoeffdingTreeWrapper()
    res_ht = evaluate_stream_prequential(ref_ht, test_stream, is_timed=True)
    print(f"  -> Always-HT Test Acc: {res_ht['accuracy_pct']}%")
    print(f"  -> Always-HT Mean Lat: {res_ht['mean_cpu_us']} us/inst")

    # 4. Reference: Always-ARF 10-Trees
    print("\n  Executing Reference Always-ARF (10 Trees) on Test Stream...")
    ref_arf10 = ARFFullWrapper(seed=42)
    res_arf10 = evaluate_stream_prequential(ref_arf10, test_stream, is_timed=True)
    print(f"  -> Always-ARF (10T) Acc: {res_arf10['accuracy_pct']}%")
    print(f"  -> Always-ARF (10T) Lat: {res_arf10['mean_cpu_us']} us/inst")

    # Save per-instance predictions
    os.makedirs("results/raw", exist_ok=True)
    df_arf3_raw = pd.DataFrame({
        "instance_idx": np.arange(1, len(test_stream) + 1),
        "y_true": res_arf3["y_trues"],
        "y_pred": res_arf3["y_preds"],
        "gate_open": res_arf3["gate_states"],
        "cpu_time_ns": res_arf3["cpu_ns"]
    })
    df_arf3_raw.to_csv("results/raw/fair_test_predictions_arf3.csv", index=False)

    df_cgt_raw = pd.DataFrame({
        "instance_idx": np.arange(1, len(test_stream) + 1),
        "y_true": res_cgt["y_trues"],
        "y_pred": res_cgt["y_preds"],
        "gate_open": res_cgt["gate_states"],
        "cpu_time_ns": res_cgt["cpu_ns"]
    })
    df_cgt_raw.to_csv("results/raw/fair_test_predictions_cgt.csv", index=False)

    # =========================================================================
    # STEP 3: RANDOM-GATE BASELINE AT RHO = RHO_CGT
    # =========================================================================
    print(f"\n[STEP 3/7] Running Random-Gate Baseline at Matching Duty Cycle (rho = {rho_cgt_empirical:.4f})...")
    random_gate = RandomGateBaseline(
        p=rho_cgt_empirical,
        arf_config=best_arf3_config,
        seed=42
    )
    res_rg = evaluate_stream_prequential(random_gate, test_stream, is_timed=True)
    print(f"  -> Random-Gate Test Acc:     {res_rg['accuracy_pct']}%")
    print(f"  -> Random-Gate Duty Cycle:   {res_rg['duty_cycle'] * 100.0:.2f}%")
    print(f"  -> Random-Gate Mean Latency: {res_rg['mean_cpu_us']} us/inst (Worst: {res_rg['latency_max_us']} us)")

    df_rg_raw = pd.DataFrame({
        "instance_idx": np.arange(1, len(test_stream) + 1),
        "y_true": res_rg["y_trues"],
        "y_pred": res_rg["y_preds"],
        "gate_open": res_rg["gate_states"],
        "cpu_time_ns": res_rg["cpu_ns"]
    })
    df_rg_raw.to_csv("results/raw/fair_test_predictions_random_gate.csv", index=False)

    # Save head-to-head summary
    test_summary_rows = [
        {
            "Model": "Always-HT (Baseline)",
            "Ensemble Size": 1,
            "Duty Cycle (%)": round(res_ht["duty_cycle"] * 100.0, 2),
            "Test Accuracy (%)": res_ht["accuracy_pct"],
            "Mean Latency (us)": res_ht["mean_cpu_us"],
            "p95 Latency (us)": res_ht["latency_p95_us"],
            "Worst-case Latency (us)": res_ht["latency_max_us"],
            "Gating Nature": "Always Closed (0% Heavy)"
        },
        {
            "Model": f"Random-Gate (p={rho_cgt_empirical:.1%})",
            "Ensemble Size": "1 / 3",
            "Duty Cycle (%)": round(res_rg["duty_cycle"] * 100.0, 2),
            "Test Accuracy (%)": res_rg["accuracy_pct"],
            "Mean Latency (us)": res_rg["mean_cpu_us"],
            "p95 Latency (us)": res_rg["latency_p95_us"],
            "Worst-case Latency (us)": res_rg["latency_max_us"],
            "Gating Nature": "Pure Bernoulli Random Escalation"
        },
        {
            "Model": "CGT (Proposed Meta-Gate)",
            "Ensemble Size": "1 / 3",
            "Duty Cycle (%)": round(res_cgt["duty_cycle"] * 100.0, 2),
            "Test Accuracy (%)": res_cgt["accuracy_pct"],
            "Mean Latency (us)": res_cgt["mean_cpu_us"],
            "p95 Latency (us)": res_cgt["latency_p95_us"],
            "Worst-case Latency (us)": res_cgt["latency_max_us"],
            "Gating Nature": "Selective Learned Margin + ADWIN Cooldown"
        },
        {
            "Model": "ARF-3 (3 Trees)",
            "Ensemble Size": 3,
            "Duty Cycle (%)": round(res_arf3["duty_cycle"] * 100.0, 2),
            "Test Accuracy (%)": res_arf3["accuracy_pct"],
            "Mean Latency (us)": res_arf3["mean_cpu_us"],
            "p95 Latency (us)": res_arf3["latency_p95_us"],
            "Worst-case Latency (us)": res_arf3["latency_max_us"],
            "Gating Nature": "Always Open (100% Heavy)"
        },
        {
            "Model": "Always-ARF (10 Trees Reference)",
            "Ensemble Size": 10,
            "Duty Cycle (%)": round(res_arf10["duty_cycle"] * 100.0, 2),
            "Test Accuracy (%)": res_arf10["accuracy_pct"],
            "Mean Latency (us)": res_arf10["mean_cpu_us"],
            "p95 Latency (us)": res_arf10["latency_p95_us"],
            "Worst-case Latency (us)": res_arf10["latency_max_us"],
            "Gating Nature": "Always Open (100% Full Ensemble)"
        }
    ]
    df_test_summary = pd.DataFrame(test_summary_rows)
    df_test_summary.to_csv("results/tables/fair_comparison_test_results.csv", index=False)
    print("\nHead-to-Head Test Summary Table:")
    print(df_test_summary.to_string(index=False))

    # =========================================================================
    # STEP 4: DUTY BUDGET SWEEP (5, 10, 15, 25, 40, 100%)
    # =========================================================================
    print("\n[STEP 4/7] Running Duty Budget Sweep (5%, 10%, 15%, 25%, 40%, 100%)...")
    budget_levels = [0.05, 0.10, 0.15, 0.25, 0.40, 1.00]
    sweep_records = []

    for b in budget_levels:
        b_pct = int(b * 100)
        print(f"  Sweeping Budget Level: {b_pct}%...")
        
        # 1. CGT with max_rho = b
        arf_h = ARF3Wrapper(
            grace_period=best_arf3_config["grace_period"],
            lambda_value=best_arf3_config["lambda_value"],
            delta=best_arf3_config["delta"],
            seed=42
        )
        cgt_sweep = CGTv3LearnedGateController(
            tau_gate=best_cgt_config["tau_gate"],
            max_rho=b,
            h_cooldown=best_cgt_config["h_cooldown"],
            buffer_b=best_cgt_config["buffer_b"],
            adwin_delta=best_cgt_config["adwin_delta"],
            arf_model=arf_h,
            seed=42
        )
        res_cgt_sw = evaluate_stream_prequential(cgt_sweep, test_stream, is_timed=False)
        sweep_records.append({
            "Policy": "CGT (Selective Learned Gate)",
            "Budget_Cap_pct": b_pct,
            "Empirical_Duty_pct": round(res_cgt_sw["duty_cycle"] * 100.0, 2),
            "Test_Accuracy_pct": res_cgt_sw["accuracy_pct"],
            "Mean_Latency_us": res_cgt_sw["mean_cpu_us"]
        })
        
        # 2. Random-Gate with p = b
        rg_sweep = RandomGateBaseline(
            p=b,
            arf_config=best_arf3_config,
            seed=42
        )
        res_rg_sw = evaluate_stream_prequential(rg_sweep, test_stream, is_timed=False)
        sweep_records.append({
            "Policy": "Random-Gate Baseline",
            "Budget_Cap_pct": b_pct,
            "Empirical_Duty_pct": round(res_rg_sw["duty_cycle"] * 100.0, 2),
            "Test_Accuracy_pct": res_rg_sw["accuracy_pct"],
            "Mean_Latency_us": res_rg_sw["mean_cpu_us"]
        })

    df_sweep = pd.DataFrame(sweep_records)
    df_sweep.to_csv("results/tables/duty_budget_sweep.csv", index=False)
    print("\nDuty Budget Sweep Results:")
    print(df_sweep.to_string(index=False))

    # Generate Publication-Quality Plot: Accuracy vs Duty Cycle
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    cgt_sw_df = df_sweep[df_sweep["Policy"] == "CGT (Selective Learned Gate)"].sort_values("Empirical_Duty_pct")
    rg_sw_df = df_sweep[df_sweep["Policy"] == "Random-Gate Baseline"].sort_values("Empirical_Duty_pct")
    
    # Plot curves
    ax.plot(
        cgt_sw_df["Empirical_Duty_pct"],
        cgt_sw_df["Test_Accuracy_pct"],
        color="#1f77b4",
        marker="o",
        linewidth=2.4,
        markersize=7,
        label="CGT (Selective Meta-Gated Routing)"
    )
    ax.plot(
        rg_sw_df["Empirical_Duty_pct"],
        rg_sw_df["Test_Accuracy_pct"],
        color="#d62728",
        marker="s",
        linestyle="--",
        linewidth=2.0,
        markersize=6,
        label="Random-Gate Baseline (Bernoulli Escalation)"
    )
    
    # Reference lines
    ax.axhline(
        res_ht["accuracy_pct"],
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.5,
        label=f"Always-HT Reference ({res_ht['accuracy_pct']}%, 0% Heavy)"
    )
    ax.axhline(
        res_arf3["accuracy_pct"],
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.5,
        label=f"Always-ARF-3 Reference ({res_arf3['accuracy_pct']}%, 100% Heavy)"
    )
    
    # Shading and annotations
    ax.fill_between(
        cgt_sw_df["Empirical_Duty_pct"],
        rg_sw_df["Test_Accuracy_pct"],
        cgt_sw_df["Test_Accuracy_pct"],
        color="#1f77b4",
        alpha=0.15,
        label="Learned Meta-Gating Value (Oracle Targeting Gain)"
    )
    
    ax.set_title("Stream Mining Trade-off: Prequential Accuracy vs. Compute Duty Budget\n(24,528-Hour European Grid Test Stream)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Empirical Heavy Execution Duty Cycle (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Prequential Classification Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(-2, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
    
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plot_path = "results/figures/fig_accuracy_vs_duty_sweep.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n  Saved sweep plot to {plot_path}")

    # =========================================================================
    # STEP 5: MODELED MULTI-RATE HARDWARE PROFILING (50 Hz, 200 Hz, 1 kHz)
    # =========================================================================
    print("\n[STEP 5/7] Computing Modeled Multi-Rate Hardware Profiles (50 Hz, 200 Hz, 1 kHz)...")
    
    # Rates
    rates = [
        {"name": "50 Hz (Standard PMU)", "freq_hz": 50.0, "t_period_us": 20000.0},
        {"name": "200 Hz (High-Speed PMU / DFR)", "freq_hz": 200.0, "t_period_us": 5000.0},
        {"name": "1 kHz (Substation RTU Deadline)", "freq_hz": 1000.0, "t_period_us": 1000.0}
    ]
    
    # Hardware specs
    # STM32H743ZI (Arm Cortex-M7 @ 480 MHz): Active 280 mW, Sleep 8.2 mW
    # CM4 (Quad Cortex-A72 @ 1.5 GHz): Active 3000 mW, Idle 850 mW
    models_to_profile = [
        ("Always-HT", res_ht),
        ("Random-Gate (rho_CGT)", res_rg),
        ("CGT (Learned Gate)", res_cgt),
        ("ARF-3 (3 Trees)", res_arf3),
        ("Always-ARF (10 Trees)", res_arf10)
    ]
    
    hardware_records = []
    
    for r in rates:
        freq = r["freq_hz"]
        t_per = r["t_period_us"]
        
        for m_name, res in models_to_profile:
            t_mean = res["mean_cpu_us"]
            t_worst = res["latency_max_us"]
            
            # CPU Duty Cycle (modeled)
            cpu_duty = (t_mean / t_per) * 100.0
            
            # Deadline overrun flags (modeled)
            mean_schedulable = bool(t_mean <= t_per)
            worst_case_guaranteed = bool(t_worst <= t_per)
            
            # STM32H743ZI Power (modeled)
            d_frac = min(cpu_duty / 100.0, 1.0)
            p_stm32_mw = d_frac * 280.0 + (1.0 - d_frac) * 8.2
            bat_stm32_100wh_h = 100.0 / (p_stm32_mw / 1000.0)
            bat_stm32_8wh_h = 8.25 / (p_stm32_mw / 1000.0)
            
            # Raspberry Pi CM4 Power (modeled)
            p_cm4_mw = d_frac * 3000.0 + (1.0 - d_frac) * 850.0
            bat_cm4_100wh_h = 100.0 / (p_cm4_mw / 1000.0)
            
            hardware_records.append({
                "Sampling_Rate": r["name"],
                "Frequency_Hz": freq,
                "Sample_Period_us": t_per,
                "Model_Name": m_name,
                "Mean_Latency_us": t_mean,
                "Worst_Case_Latency_us": t_worst,
                "Modeled_CPU_Duty_pct": round(cpu_duty, 3),
                "Schedulable_Mean (modeled)": "YES" if mean_schedulable else "OVERRUN",
                "Hard_Deadline_Met_Worst (modeled)": "GUARANTEED" if worst_case_guaranteed else "OVERRUN",
                "STM32H743_Power_mW (modeled)": round(p_stm32_mw, 2),
                "STM32H743_100Wh_Autonomy_hrs (modeled)": round(bat_stm32_100wh_h, 1),
                "STM32H743_8.25Wh_Autonomy_hrs (modeled)": round(bat_stm32_8wh_h, 1),
                "CM4_Power_mW (modeled)": round(p_cm4_mw, 1),
                "CM4_100Wh_Autonomy_hrs (modeled)": round(bat_cm4_100wh_h, 1)
            })

    df_hw = pd.DataFrame(hardware_records)
    df_hw.to_csv("results/tables/modeled_hardware_multirate_comparison.csv", index=False)
    print("\nModeled Multi-Rate Hardware Profile Table:")
    print(df_hw[["Sampling_Rate", "Model_Name", "Mean_Latency_us", "Modeled_CPU_Duty_pct", "Hard_Deadline_Met_Worst (modeled)", "STM32H743_100Wh_Autonomy_hrs (modeled)", "CM4_100Wh_Autonomy_hrs (modeled)"]].to_string(index=False))

    # =========================================================================
    # STEP 6: CONCEPT DRIFT WINDOW DISAGGREGATION
    # =========================================================================
    print("\n[STEP 6/7] Disaggregating Accuracy Inside vs. Outside Concept Drift Windows...")
    
    # Identify drift windows on the test stream:
    # 1. ADWIN Statistical Drift Alarms on Base Residuals
    adwin_detector = ADWIN(delta=0.005, clock=32)
    is_adwin_drift_window = np.zeros(len(test_stream), dtype=bool)
    
    ht_ref_drift = HoeffdingTreeWrapper()
    drift_cooldown = 0
    drift_alarm_count = 0
    
    for idx, (x_t, y_t, ts) in enumerate(test_stream):
        p = ht_ref_drift.predict_one(x_t)
        err = 1.0 if p != y_t else 0.0
        adwin_detector.update(err)
        if adwin_detector.drift_detected:
            drift_alarm_count += 1
            drift_cooldown = 100  # 100 hours post-alarm adaptation window
            
        if drift_cooldown > 0:
            is_adwin_drift_window[idx] = True
            drift_cooldown -= 1
        ht_ref_drift.learn_one(x_t, y_t)

    # 2. Known Grid Shock Windows (European Cold Snap Jan 2017 & Beast from the East Feb/Mar 2018)
    is_physical_shock_window = np.zeros(len(test_stream), dtype=bool)
    for idx, (x_t, y_t, ts) in enumerate(test_stream):
        # Jan 15 - Feb 15 2017
        is_cold_2017 = (ts.year == 2017 and ts.month == 1 and ts.day >= 15) or (ts.year == 2017 and ts.month == 2 and ts.day <= 15)
        # Feb 24 - Mar 8 2018
        is_cold_2018 = (ts.year == 2018 and ts.month == 2 and ts.day >= 24) or (ts.year == 2018 and ts.month == 3 and ts.day <= 8)
        # Severe hourly ramp (|demand_diff_1_2| > 2500 MW, ~ 2.5 sigma)
        is_ramp = abs(x_t.get('demand_diff_1_2', 0.0)) > 2500.0
        
        if is_cold_2017 or is_cold_2018 or is_ramp:
            is_physical_shock_window[idx] = True

    # Combined Drift Regime: True if either statistical or physical shock
    is_drift_regime = is_adwin_drift_window | is_physical_shock_window
    n_drift = int(np.sum(is_drift_regime))
    n_stationary = len(test_stream) - n_drift
    
    print(f"  Drift Regime: {n_drift} hours ({n_drift / len(test_stream) * 100:.2f}% of test stream)")
    print(f"  Stationary Regime: {n_stationary} hours ({n_stationary / len(test_stream) * 100:.2f}% of test stream)")

    drift_records = []
    models_to_test = [
        ("Always-HT", res_ht),
        ("Random-Gate", res_rg),
        ("CGT (Learned Gate)", res_cgt),
        ("ARF-3 (3 Trees)", res_arf3),
        ("Always-ARF (10 Trees)", res_arf10)
    ]
    
    for m_name, res in models_to_test:
        correct_arr = (res["y_preds"] == res["y_trues"]).astype(float)
        
        acc_overall = float(np.mean(correct_arr)) * 100.0
        acc_inside = float(np.mean(correct_arr[is_drift_regime])) * 100.0
        acc_outside = float(np.mean(correct_arr[~is_drift_regime])) * 100.0
        delta_drift = acc_inside - acc_outside
        
        # Duty cycle inside vs outside
        gate_arr = res["gate_states"].astype(float)
        duty_inside = float(np.mean(gate_arr[is_drift_regime])) * 100.0
        duty_outside = float(np.mean(gate_arr[~is_drift_regime])) * 100.0
        
        drift_records.append({
            "Model_Name": m_name,
            "Overall_Accuracy_pct": round(acc_overall, 2),
            "Inside_Drift_Accuracy_pct": round(acc_inside, 2),
            "Outside_Drift_Accuracy_pct": round(acc_outside, 2),
            "Accuracy_Delta_pct (Inside - Outside)": round(delta_drift, 2),
            "Duty_Cycle_Inside_Drift_pct": round(duty_inside, 2),
            "Duty_Cycle_Outside_Drift_pct": round(duty_outside, 2)
        })

    df_drift = pd.DataFrame(drift_records)
    df_drift.to_csv("results/tables/drift_window_accuracy_comparison.csv", index=False)
    print("\nConcept Drift Window Disaggregation Table:")
    print(df_drift.to_string(index=False))

    # =========================================================================
    # STEP 7: OUTCOME CHECK & TRANSPARENT REPORTING
    # =========================================================================
    print("\n" + "=" * 80)
    print("FAIR COMPARISON FINAL VERDICT & REPORTING")
    print("=" * 80)
    
    arf3_acc = res_arf3["accuracy_pct"]
    cgt_acc = res_cgt["accuracy_pct"]
    rg_acc = res_rg["accuracy_pct"]
    
    print(f"ARF-3 Test Accuracy:       {arf3_acc:.3f}% (Duty = 100.0%, Mean Latency = {res_arf3['mean_cpu_us']} us)")
    print(f"CGT Test Accuracy:         {cgt_acc:.3f}% (Duty = {res_cgt['duty_cycle']*100:.2f}%, Mean Latency = {res_cgt['mean_cpu_us']} us)")
    print(f"Random-Gate Test Accuracy: {rg_acc:.3f}% (Duty = {res_rg['duty_cycle']*100:.2f}%, Mean Latency = {res_rg['mean_cpu_us']} us)")
    print(f"Accuracy Delta (ARF-3 - CGT): {arf3_acc - cgt_acc:+.3f} percentage points")
    print(f"Accuracy Delta (CGT - Random-Gate): {cgt_acc - rg_acc:+.3f} percentage points")
    print("-" * 80)

    arf3_wins = bool(arf3_acc > cgt_acc)
    
    if arf3_wins:
        print(">>> OUTCOME: ARF-3 WINS on raw prequential test accuracy!")
        print(">>> IN ACCORDANCE WITH USER INSTRUCTION: 'if arf 3 wins just inform and if it does not upload to github'")
        print(">>> GITHUB UPLOAD SUPPRESSED. Providing full transparent report directly to user.")
    else:
        print(">>> OUTCOME: CGT WINS or matches ARF-3 on raw test accuracy!")
        print(">>> Ready for GitHub synchronization.")
        
    print("=" * 80)
    return {
        "arf3_wins": arf3_wins,
        "arf3_acc": arf3_acc,
        "cgt_acc": cgt_acc,
        "rg_acc": rg_acc,
        "frozen_hash": frozen_hash,
        "df_test_summary": df_test_summary,
        "df_hw": df_hw,
        "df_drift": df_drift,
        "df_sweep": df_sweep
    }


if __name__ == "__main__":
    main()
