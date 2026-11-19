"""
CGT-v4 Master Comparative Benchmark Across Both Data Streams
=============================================================================
LABEL: post-hoc: developed after viewing test results
=============================================================================

Evaluates:
1. Streaming Linear / Logistic Regression (SGD)
2. Always-HT (Single Tree Baseline)
3. Random-Gate Baseline (20 seeds mean +/- std)
4. Cost-Aware OMS-MAB (Wilson et al., matched duty)
5. CGT-v3 (Pre-hoc Learned Meta-Gate)
6. CGT-v4 (Post-hoc, ARF-3 Heavy Arm, Amortized/Standby)
7. CGT-v4 (Post-hoc, ARF-10 Heavy Arm)
8. ARF-3 (3 Trees, Continuous 100% Duty)
9. Always-ARF (10 Trees, Continuous 100% Duty)

Across:
- Stream 1: Primary SCADA Benchmark Stream (24,528 continuous test hours / 146 weekly blocks)
- Stream 2: Real POSOCO Indian National Grid SCADA Stream (16,800 continuous test hours / 100 weekly blocks)

Captures:
- Accuracy (%), Mean Latency (us), p50 (us), p95 (us), p99 (us), Max Latency (us), Duty Cycle (rho)
- Weekly block paired Wilcoxon tests with Holm-Bonferroni correction
- Plots Accuracy vs Mean Latency for BOTH streams.
"""

import os
import sys
import time
import json
import hashlib
import tempfile
import numpy as np
import pandas as pd
from scipy import stats

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from river.linear_model import LogisticRegression
from river.preprocessing import StandardScaler
from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper
from src.controllers.cgt_v3 import CGTv3LearnedGateController
from src.controllers.cgt_v4 import CGTv4Controller
from scripts.run_real_posoco_benchmark import load_real_posoco_stream, CostAwareOMSMABController


class LinearSGDWrapper:
    """Online streaming standard scaler + logistic regression with SGD."""
    def __init__(self, lr=0.01):
        self.model = StandardScaler() | LogisticRegression(optimizer=None) # river default SGD
        self.name = "Streaming Linear (SGD)"
        self.gate_open = False
        
    def predict_one(self, x):
        proba = self.model.predict_proba_one(x)
        p0 = proba.get(0, 0.5)
        p1 = proba.get(1, 0.5)
        return 1 if p1 > p0 else 0
        
    def learn_one(self, x, y, y_pred=None):
        self.model.learn_one(x, y)


class ARFWrapper:
    """Standardized ARF wrapper."""
    def __init__(self, n_models=3, grace_period=50, lambda_value=6, delta=0.01, seed=42):
        self.n_models = n_models
        self.name = f"ARF-{n_models} ({n_models} Trees)"
        self.model = ARFClassifier(
            n_models=n_models,
            grace_period=grace_period,
            lambda_value=lambda_value,
            delta=delta,
            split_criterion='info_gain',
            seed=seed
        )
        self.gate_open = True

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y, y_pred=None):
        self.model.learn_one(x, y)


class RandomGateMultiSeedWrapper:
    """Evaluates Random Gate across 20 random seeds."""
    def __init__(self, p=0.046, n_seeds=20):
        self.p = p
        self.n_seeds = n_seeds
        self.name = f"Random-Gate ({n_seeds} seeds, p={p:.2%})"
        self.gate_open = False


def evaluate_stream_detailed(model, stream):
    """Evaluates a model over a stream with per-instance microsecond latency recording."""
    n = len(stream)
    y_trues = np.zeros(n, dtype=int)
    y_preds = np.zeros(n, dtype=int)
    gate_states = np.zeros(n, dtype=int)
    latencies_ns = np.zeros(n, dtype=np.int64)
    
    correct = 0
    t_start = time.perf_counter_ns()
    
    for idx, item in enumerate(stream):
        x_t, y_t = item[0], item[1]
        y_trues[idx] = y_t
        t0 = time.perf_counter_ns()
        
        pred = model.predict_one(x_t)
        if hasattr(model, 'learn_one'):
            try:
                model.learn_one(x_t, y_t, pred)
            except TypeError:
                model.learn_one(x_t, y_t)
                
        latencies_ns[idx] = time.perf_counter_ns() - t0
        y_preds[idx] = pred
        
        is_gate = 1 if getattr(model, 'gate_open', False) else (1 if isinstance(model, ARFWrapper) else 0)
        gate_states[idx] = is_gate
        
        if pred == y_t:
            correct += 1
            
    t_total_ns = time.perf_counter_ns() - t_start
    acc = (correct / n) * 100.0
    mean_cpu_us = (t_total_ns / n) / 1000.0
    duty = float(np.mean(gate_states))
    
    return {
        "accuracy_pct": round(acc, 3),
        "mean_cpu_us": round(mean_cpu_us, 2),
        "p50_us": round(float(np.percentile(latencies_ns, 50)) / 1000.0, 2),
        "p95_us": round(float(np.percentile(latencies_ns, 95)) / 1000.0, 2),
        "p99_us": round(float(np.percentile(latencies_ns, 99)) / 1000.0, 2),
        "max_us": round(float(np.max(latencies_ns)) / 1000.0, 2),
        "duty_cycle": round(duty, 4),
        "correct_count": correct,
        "total_instances": n,
        "y_trues": y_trues,
        "y_preds": y_preds,
        "gate_states": gate_states
    }


def evaluate_random_gate_20seeds(stream, p=0.046, n_seeds=20):
    accs = []
    means = []
    p50s, p95s, p99s, maxs = [], [], [], []
    duties = []
    seed_preds = []
    
    for s in range(n_seeds):
        rng = np.random.RandomState(42 + s)
        ht = HoeffdingTreeWrapper()
        arf = ARFWrapper(n_models=3, seed=42 + s)
        
        n = len(stream)
        correct = 0
        latencies = np.zeros(n, dtype=np.int64)
        gate_states = np.zeros(n, dtype=int)
        y_preds = np.zeros(n, dtype=int)
        
        for idx, item in enumerate(stream):
            x_t, y_t = item[0], item[1]
            t0 = time.perf_counter_ns()
            open_gate = bool(rng.rand() < p)
            gate_states[idx] = 1 if open_gate else 0
            
            if open_gate:
                pred = arf.predict_one(x_t)
            else:
                pred = ht.predict_one(x_t)
                
            ht.learn_one(x_t, y_t)
            if open_gate:
                arf.learn_one(x_t, y_t)
                
            latencies[idx] = time.perf_counter_ns() - t0
            y_preds[idx] = pred
            if pred == y_t:
                correct += 1
                
        accs.append((correct / n) * 100.0)
        means.append(float(np.mean(latencies)) / 1000.0)
        p50s.append(float(np.percentile(latencies, 50)) / 1000.0)
        p95s.append(float(np.percentile(latencies, 95)) / 1000.0)
        p99s.append(float(np.percentile(latencies, 99)) / 1000.0)
        maxs.append(float(np.max(latencies)) / 1000.0)
        duties.append(float(np.mean(gate_states)))
        seed_preds.append(y_preds)
        
    avg_preds = np.round(np.mean(seed_preds, axis=0)).astype(int)
    
    return {
        "accuracy_pct": round(float(np.mean(accs)), 3),
        "accuracy_std": round(float(np.std(accs)), 3),
        "mean_cpu_us": round(float(np.mean(means)), 2),
        "p50_us": round(float(np.mean(p50s)), 2),
        "p95_us": round(float(np.mean(p95s)), 2),
        "p99_us": round(float(np.mean(p99s)), 2),
        "max_us": round(float(np.max(maxs)), 2),
        "duty_cycle": round(float(np.mean(duties)), 4),
        "y_preds": avg_preds
    }


def compute_weekly_blocks(y_trues, preds_dict, block_size=168):
    n = len(y_trues)
    n_blocks = n // block_size
    trimmed_n = n_blocks * block_size
    
    block_accs = {name: [] for name in preds_dict}
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        y_true_b = y_trues[start:end]
        
        for name, p in preds_dict.items():
            acc_b = np.mean(p[start:end] == y_true_b) * 100.0
            block_accs[name].append(acc_b)
            
    return pd.DataFrame(block_accs)


def main():
    print("=" * 80)
    print("CGT-v4 MASTER COMPARATIVE BENCHMARK ACROSS BOTH STREAMS")
    print("LABEL: post-hoc: developed after viewing test results")
    print("=" * 80)
    
    # Load frozen CGT-v4 configuration
    cfg_path = "configs/cgt_v4_frozen_config.json"
    with open(cfg_path, "r") as f:
        frozen_data = json.load(f)
    cfg_amortized = frozen_data["cgt_v4_amortized_arf3"]
    cfg_standby3 = frozen_data["cgt_v4_standby_arf3"]
    cfg_standby10 = frozen_data["cgt_v4_standby_arf10"]
    print(f"Loaded CGT-v4 frozen configs (SHA-256: {frozen_data['metadata']['sha256'][:16]}...)")
    
    # -------------------------------------------------------------------------
    # PART 1: STREAM 1 - PRIMARY SCADA BENCHMARK STREAM (24,528 TEST HOURS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATING ON STREAM 1: PRIMARY SCADA BENCHMARK STREAM (N = 24,528)")
    print("=" * 80)
    
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_instances = 10512
    test_instances = 24528
    test_stream1 = full_stream[val_instances:val_instances + test_instances]
    y_trues_stream1 = np.array([item[1] for item in test_stream1])
    
    models_stream1 = {
        "Linear (SGD)": LinearSGDWrapper(),
        "Always-HT": HoeffdingTreeWrapper(),
        "Cost-Aware OMS-MAB": CostAwareOMSMABController(lambda_cost=0.01, max_rho=0.046, seed=42),
        "CGT-v3 (Pre-hoc)": CGTv3LearnedGateController(tau_gate=0.25, max_rho=0.12, h_cooldown=50, seed=42),
        "CGT-v4 (Amortized, ARF-3)": CGTv4Controller(
            tau_gate=cfg_amortized["tau_gate"],
            max_rho=cfg_amortized["max_rho"],
            h_cooldown=cfg_amortized["h_cooldown"],
            replay_mode=cfg_amortized["replay_mode"],
            buffer_b=cfg_amortized["buffer_b"],
            k_replay_per_step=cfg_amortized["k_replay_per_step"],
            batch_size=cfg_amortized["batch_size"],
            n_heavy_trees=3,
            seed=42
        ),
        "CGT-v4 (Warm Standby, ARF-3)": CGTv4Controller(
            tau_gate=cfg_standby3["tau_gate"],
            max_rho=cfg_standby3["max_rho"],
            h_cooldown=cfg_standby3["h_cooldown"],
            replay_mode=cfg_standby3["replay_mode"],
            buffer_b=cfg_standby3["buffer_b"],
            k_replay_per_step=cfg_standby3["k_replay_per_step"],
            batch_size=cfg_standby3["batch_size"],
            n_heavy_trees=3,
            seed=42
        ),
        "CGT-v4 (Warm Standby, ARF-10)": CGTv4Controller(
            tau_gate=cfg_standby10["tau_gate"],
            max_rho=cfg_standby10["max_rho"],
            h_cooldown=cfg_standby10["h_cooldown"],
            replay_mode=cfg_standby10["replay_mode"],
            buffer_b=cfg_standby10["buffer_b"],
            k_replay_per_step=cfg_standby10["k_replay_per_step"],
            batch_size=cfg_standby10["batch_size"],
            n_heavy_trees=10,
            seed=42
        ),
        "ARF-3 (Fair Protocol)": ARFWrapper(n_models=3, seed=42),
        "Always-ARF (10 Trees)": ARFWrapper(n_models=10, seed=42)
    }
    
    results_s1 = {}
    preds_s1 = {}
    
    for name, m in models_stream1.items():
        print(f"Evaluating {name} on Stream 1...", end=" ", flush=True)
        res = evaluate_stream_detailed(m, test_stream1)
        results_s1[name] = res
        preds_s1[name] = res["y_preds"]
        print(f"-> Acc: {res['accuracy_pct']}%, Mean: {res['mean_cpu_us']} us, p99: {res['p99_us']} us, Max: {res['max_us']} us, Duty: {res['duty_cycle']*100:.2f}%")
        
    print("Evaluating Random-Gate (20 seeds) on Stream 1...", end=" ", flush=True)
    res_rg1 = evaluate_random_gate_20seeds(test_stream1, p=results_s1["CGT-v4 (Amortized, ARF-3)"]["duty_cycle"], n_seeds=20)
    results_s1["Random-Gate (20 seeds)"] = res_rg1
    preds_s1["Random-Gate (20 seeds)"] = res_rg1["y_preds"]
    print(f"-> Acc: {res_rg1['accuracy_pct']} +/- {res_rg1['accuracy_std']}%, Mean: {res_rg1['mean_cpu_us']} us, Duty: {res_rg1['duty_cycle']*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # PART 2: STREAM 2 - REAL POSOCO SCADA STREAM (16,800 TEST HOURS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATING ON STREAM 2: REAL POSOCO INDIAN GRID SCADA (N = 16,800)")
    print("=" * 80)
    
    full_posoco_stream = load_real_posoco_stream("data/india_electricity_real_hourly.csv", warmup_rows=24)
    posoco_val_len = 3624
    posoco_test_len = 16800
    test_stream2 = full_posoco_stream[posoco_val_len:posoco_val_len + posoco_test_len]
    y_trues_stream2 = np.array([item[1] for item in test_stream2])
    print(f"POSOCO Test Stream Length: {len(test_stream2)} hours ({len(test_stream2)//168} weekly blocks)")
    
    models_stream2 = {
        "Linear (SGD)": LinearSGDWrapper(),
        "Always-HT": HoeffdingTreeWrapper(),
        "Cost-Aware OMS-MAB": CostAwareOMSMABController(lambda_cost=0.01, max_rho=0.058, seed=42),
        "CGT-v3 (Pre-hoc)": CGTv3LearnedGateController(tau_gate=0.25, max_rho=0.12, h_cooldown=50, seed=42),
        "CGT-v4 (Amortized, ARF-3)": CGTv4Controller(
            tau_gate=cfg_amortized["tau_gate"],
            max_rho=cfg_amortized["max_rho"],
            h_cooldown=cfg_amortized["h_cooldown"],
            replay_mode=cfg_amortized["replay_mode"],
            buffer_b=cfg_amortized["buffer_b"],
            k_replay_per_step=cfg_amortized["k_replay_per_step"],
            batch_size=cfg_amortized["batch_size"],
            n_heavy_trees=3,
            seed=42
        ),
        "CGT-v4 (Warm Standby, ARF-3)": CGTv4Controller(
            tau_gate=cfg_standby3["tau_gate"],
            max_rho=cfg_standby3["max_rho"],
            h_cooldown=cfg_standby3["h_cooldown"],
            replay_mode=cfg_standby3["replay_mode"],
            buffer_b=cfg_standby3["buffer_b"],
            k_replay_per_step=cfg_standby3["k_replay_per_step"],
            batch_size=cfg_standby3["batch_size"],
            n_heavy_trees=3,
            seed=42
        ),
        "CGT-v4 (Warm Standby, ARF-10)": CGTv4Controller(
            tau_gate=cfg_standby10["tau_gate"],
            max_rho=cfg_standby10["max_rho"],
            h_cooldown=cfg_standby10["h_cooldown"],
            replay_mode=cfg_standby10["replay_mode"],
            buffer_b=cfg_standby10["buffer_b"],
            k_replay_per_step=cfg_standby10["k_replay_per_step"],
            batch_size=cfg_standby10["batch_size"],
            n_heavy_trees=10,
            seed=42
        ),
        "ARF-3 (Fair Protocol)": ARFWrapper(n_models=3, seed=42),
        "Always-ARF (10 Trees)": ARFWrapper(n_models=10, seed=42)
    }
    
    results_s2 = {}
    preds_s2 = {}
    
    for name, m in models_stream2.items():
        print(f"Evaluating {name} on Stream 2 (POSOCO)...", end=" ", flush=True)
        res = evaluate_stream_detailed(m, test_stream2)
        results_s2[name] = res
        preds_s2[name] = res["y_preds"]
        print(f"-> Acc: {res['accuracy_pct']}%, Mean: {res['mean_cpu_us']} us, p99: {res['p99_us']} us, Max: {res['max_us']} us, Duty: {res['duty_cycle']*100:.2f}%")
        
    print("Evaluating Random-Gate (20 seeds) on Stream 2 (POSOCO)...", end=" ", flush=True)
    res_rg2 = evaluate_random_gate_20seeds(test_stream2, p=results_s2["CGT-v4 (Amortized, ARF-3)"]["duty_cycle"], n_seeds=20)
    results_s2["Random-Gate (20 seeds)"] = res_rg2
    preds_s2["Random-Gate (20 seeds)"] = res_rg2["y_preds"]
    print(f"-> Acc: {res_rg2['accuracy_pct']} +/- {res_rg2['accuracy_std']}%, Mean: {res_rg2['mean_cpu_us']} us, Duty: {res_rg2['duty_cycle']*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # PART 3: MASTER BENCHMARK TABLE CREATION
    # -------------------------------------------------------------------------
    model_order = [
        "Linear (SGD)",
        "Always-HT",
        "Random-Gate (20 seeds)",
        "Cost-Aware OMS-MAB",
        "CGT-v3 (Pre-hoc)",
        "CGT-v4 (Amortized, ARF-3)",
        "CGT-v4 (Warm Standby, ARF-3)",
        "CGT-v4 (Warm Standby, ARF-10)",
        "ARF-3 (Fair Protocol)",
        "Always-ARF (10 Trees)"
    ]
    
    master_records = []
    for m_name in model_order:
        r1 = results_s1[m_name]
        r2 = results_s2[m_name]
        master_records.append({
            "Model": m_name,
            "Label": "Post-hoc: developed after viewing test results" if "CGT-v4" in m_name else "Pre-hoc",
            "Stream1_Acc_Pct": r1["accuracy_pct"],
            "Stream1_Mean_us": r1["mean_cpu_us"],
            "Stream1_p99_us": r1["p99_us"],
            "Stream1_Max_us": r1["max_us"],
            "Stream1_Duty_Pct": round(r1["duty_cycle"] * 100.0, 2),
            "Stream2_POSOCO_Acc_Pct": r2["accuracy_pct"],
            "Stream2_POSOCO_Mean_us": r2["mean_cpu_us"],
            "Stream2_POSOCO_p99_us": r2["p99_us"],
            "Stream2_POSOCO_Max_us": r2["max_us"],
            "Stream2_POSOCO_Duty_Pct": round(r2["duty_cycle"] * 100.0, 2)
        })
        
    df_master = pd.DataFrame(master_records)
    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    df_master.to_csv(os.path.join(out_dir, "cgt_v4_master_benchmark.csv"), index=False)
    
    print("\n" + "=" * 80)
    print("MASTER COMPARATIVE BENCHMARK TABLE (results/tables/cgt_v4_master_benchmark.csv):")
    print("=" * 80)
    print(df_master.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # PART 4: WEEKLY BLOCK HYPOTHESIS TESTS (HOLM-BONFERRONI)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("WEEKLY BLOCK HYPOTHESIS TESTING (HOLM-BONFERRONI CORRECTED)")
    print("=" * 80)
    
    df_blocks_s1 = compute_weekly_blocks(y_trues_stream1, preds_s1, block_size=168)
    df_blocks_s2 = compute_weekly_blocks(y_trues_stream2, preds_s2, block_size=168)
    
    block_test_records = []
    
    def run_paired_tests(df_blocks, stream_name, ref_model="CGT-v4 (Amortized, ARF-3)"):
        ref_series = df_blocks[ref_model].values
        comparisons = [m for m in df_blocks.columns if m != ref_model]
        
        raw_p_values = []
        comp_records = []
        for comp in comparisons:
            comp_series = df_blocks[comp].values
            diff = ref_series - comp_series
            mean_delta = float(np.mean(diff))
            
            # Wilcoxon signed rank test
            # Handle zero differences
            non_zero_diff = diff[diff != 0]
            if len(non_zero_diff) > 0:
                stat_w, p_val = stats.wilcoxon(diff, alternative='two-sided')
            else:
                stat_w, p_val = 0.0, 1.0
                
            cohen_d = mean_delta / (float(np.std(diff)) + 1e-8)
            raw_p_values.append(p_val)
            comp_records.append({
                "Stream": stream_name,
                "Reference": ref_model,
                "Comparison": comp,
                "Mean_Weekly_Delta_Pct": round(mean_delta, 3),
                "Wilcoxon_W": round(float(stat_w), 1),
                "Raw_p_value": p_val,
                "Cohen_d": round(cohen_d, 3)
            })
            
        # Holm-Bonferroni correction
        m_tests = len(raw_p_values)
        sorted_indices = np.argsort(raw_p_values)
        holm_p_values = [0.0] * m_tests
        
        running_max = 0.0
        for rank, idx in enumerate(sorted_indices):
            raw_p = raw_p_values[idx]
            multiplier = m_tests - rank
            adj_p = min(1.0, raw_p * multiplier)
            running_max = max(running_max, adj_p)
            holm_p_values[idx] = min(1.0, running_max)
            
        for idx, rec in enumerate(comp_records):
            rec["Holm_adj_p_value"] = holm_p_values[idx]
            rec["Significant_at_05"] = holm_p_values[idx] < 0.05
            block_test_records.append(rec)
            
    run_paired_tests(df_blocks_s1, "Stream 1 (Benchmark 24k)")
    run_paired_tests(df_blocks_s2, "Stream 2 (POSOCO 16.8k)")
    
    df_block_tests = pd.DataFrame(block_test_records)
    df_block_tests.to_csv(os.path.join(out_dir, "cgt_v4_weekly_block_tests.csv"), index=False)
    print("\nWEEKLY BLOCK HYPOTHESIS TESTS SUMMARY:")
    print(df_block_tests[["Stream", "Comparison", "Mean_Weekly_Delta_Pct", "Raw_p_value", "Holm_adj_p_value", "Significant_at_05"]].to_string(index=False))
    
    # -------------------------------------------------------------------------
    # PART 5: PUBLICATION PLOTS - ACCURACY VS MEAN LATENCY ON BOTH STREAMS
    # -------------------------------------------------------------------------
    print("\nGenerating publication figures: Accuracy vs Mean Latency on Both Streams...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    
    color_map = {
        "Linear (SGD)": "#7f7f7f",
        "Always-HT": "#1f77b4",
        "Random-Gate (20 seeds)": "#8c564b",
        "Cost-Aware OMS-MAB": "#ff7f0e",
        "CGT-v3 (Pre-hoc)": "#9467bd",
        "CGT-v4 (Amortized, ARF-3)": "#2ca02c",
        "CGT-v4 (Warm Standby, ARF-3)": "#006400",
        "CGT-v4 (Warm Standby, ARF-10)": "#17becf",
        "ARF-3 (Fair Protocol)": "#d62728",
        "Always-ARF (10 Trees)": "#e377c2"
    }
    
    marker_map = {
        "Linear (SGD)": "X",
        "Always-HT": "o",
        "Random-Gate (20 seeds)": "v",
        "Cost-Aware OMS-MAB": "s",
        "CGT-v3 (Pre-hoc)": "D",
        "CGT-v4 (Amortized, ARF-3)": "*",
        "CGT-v4 (Warm Standby, ARF-3)": "p",
        "CGT-v4 (Warm Standby, ARF-10)": "P",
        "ARF-3 (Fair Protocol)": "^",
        "Always-ARF (10 Trees)": "h"
    }
    
    # Stream 1 Plot
    for m_name in model_order:
        r = results_s1[m_name]
        x = r["mean_cpu_us"]
        y = r["accuracy_pct"]
        c = color_map[m_name]
        mk = marker_map[m_name]
        sz = 140 if "CGT-v4" in m_name else 90
        ax1.scatter(x, y, color=c, marker=mk, s=sz, label=m_name, zorder=5, edgecolors='black', alpha=0.9)
        
    ax1.axvline(1000.0, color='crimson', linestyle='--', linewidth=1.5, alpha=0.7, label='1.0 ms RTOS Deadline')
    ax1.set_xscale('log')
    ax1.set_title("Stream 1: Primary SCADA Benchmark (24,528 h)\nAccuracy vs Mean Host Latency", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Mean Inference Latency (µs, Log Scale)", fontsize=11)
    ax1.set_ylabel("Prequential Test Accuracy (%)", fontsize=11)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    
    # Stream 2 Plot (POSOCO)
    for m_name in model_order:
        r = results_s2[m_name]
        x = r["mean_cpu_us"]
        y = r["accuracy_pct"]
        c = color_map[m_name]
        mk = marker_map[m_name]
        sz = 140 if "CGT-v4" in m_name else 90
        ax2.scatter(x, y, color=c, marker=mk, s=sz, label=m_name, zorder=5, edgecolors='black', alpha=0.9)
        
    ax2.axvline(1000.0, color='crimson', linestyle='--', linewidth=1.5, alpha=0.7, label='1.0 ms RTOS Deadline')
    ax2.set_xscale('log')
    ax2.set_title("Stream 2: Real POSOCO Indian Grid SCADA (16,800 h)\nAccuracy vs Mean Host Latency", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Mean Inference Latency (µs, Log Scale)", fontsize=11)
    ax2.set_ylabel("Prequential Test Accuracy (%)", fontsize=11)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    fig_dir = "results/figures"
    os.makedirs(fig_dir, exist_ok=True)
    out_fig = os.path.join(fig_dir, "fig_cgt_v4_accuracy_vs_latency_both_streams.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved figure: {out_fig}")
    print("\n" + "=" * 80)
    print("CGT-v4 MASTER BENCHMARK PIPELINE COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()

