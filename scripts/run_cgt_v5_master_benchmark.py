"""
CGT-v5 Master Comparative Benchmark Across Both Data Streams
=============================================================================
LABEL: post-hoc: developed after viewing test results
=============================================================================

Compares:
1. Linear (SGD) [Fast Linear Baseline]
2. Always-HT [Single Tree Fast Baseline]
3. Always-ARF (10 Trees) [Heavy Baseline Reference, breaks 1ms budget]
4. ARF-3 (3 Trees, Continuous 100% Duty) [Fair SOTA Baseline]
5. CGT-v4 (Amortized, ARF-3) [Previous Best Gated Controller]
6. CGT-v5 (Calibrated Cascade, ARF-10, Gated Only) [New Epistemic Controller]
7. CGT-v5 (Calibrated Cascade, ARF-10, Subsampled Bg=4) [New Anti-Staleness Controller]

Across:
- Stream 1: Primary Industrial SCADA Stream (24,528 test hours, 146 weekly blocks)
- Stream 2: Real POSOCO Indian National Grid SCADA Stream (16,800 test hours, 100 weekly blocks)
"""

import os
import sys
import time
import json
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
from src.controllers.cgt_v4 import CGTv4Controller
from src.controllers.cgt_v5 import CGTv5CalibratedController
from scripts.run_real_posoco_benchmark import load_real_posoco_stream, CostAwareOMSMABController


class LinearSGDWrapper:
    """Online streaming standard scaler + logistic regression with SGD."""
    def __init__(self, lr=0.01):
        self.model = StandardScaler() | LogisticRegression(optimizer=None)
        self.name = "Streaming Linear (SGD)"
        self.gate_open = False

    def predict_one(self, x):
        p = self.model.predict_proba_one(x)
        p0 = p.get(0, 0.5)
        p1 = p.get(1, 0.5)
        return 1 if p1 > p0 else 0

    def learn_one(self, x, y, y_pred=None):
        self.model.learn_one(x, y)


class ARFWrapper:
    def __init__(self, n_models=3, seed=42):
        self.model = ARFClassifier(
            n_models=n_models,
            grace_period=50,
            lambda_value=6,
            delta=0.01,
            split_criterion='info_gain',
            seed=seed
        )
        self.name = f"ARF-{n_models}"
        self.gate_open = True

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y, y_pred=None):
        self.model.learn_one(x, y)


def evaluate_stream_detailed(model, stream):
    n = len(stream)
    correct = 0
    latencies_ns = np.zeros(n, dtype=np.int64)
    gate_states = np.zeros(n, dtype=int)
    y_preds = np.zeros(n, dtype=int)
    y_trues = np.zeros(n, dtype=int)

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


def compute_weekly_blocks(y_trues, preds_dict, block_size=168):
    n = len(y_trues)
    n_blocks = n // block_size
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
    print("CGT-v5 MASTER COMPARATIVE BENCHMARK ACROSS BOTH STREAMS")
    print("LABEL: post-hoc: developed after viewing test results")
    print("=" * 80)

    # 1. Load Streams
    # Stream 1
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    full_s1 = list(loader.iter_stream(warmup_rows=24))
    val_len_s1 = 10512
    test_len_s1 = 24528
    test_stream1 = full_s1[val_len_s1:val_len_s1 + test_len_s1]
    y_trues_s1 = np.array([item[1] for item in test_stream1])
    print(f"Loaded Stream 1: {len(test_stream1)} test hours ({len(test_stream1)//168} weekly blocks)")

    # Stream 2 (POSOCO)
    full_posoco_stream = load_real_posoco_stream("data/india_electricity_real_hourly.csv", warmup_rows=24)
    posoco_val_len = 3624
    posoco_test_len = 16800
    test_stream2 = full_posoco_stream[posoco_val_len:posoco_val_len + posoco_test_len]
    y_trues_s2 = np.array([item[1] for item in test_stream2])
    print(f"Loaded Stream 2 (POSOCO): {len(test_stream2)} test hours ({len(test_stream2)//168} weekly blocks)")

    # Build model dictionary
    def get_models():
        return {
            "Linear (SGD)": LinearSGDWrapper(),
            "Always-HT": HoeffdingTreeWrapper(),
            "Always-ARF (10 Trees)": ARFWrapper(n_models=10, seed=42),
            "ARF-3 (Fair Protocol)": ARFWrapper(n_models=3, seed=42),
            "CGT-v4 (Amortized, ARF-3)": CGTv4Controller(tau_gate=0.25, max_rho=0.12, buffer_b=50, k_replay_per_step=5, n_heavy_trees=3, seed=42),
            "CGT-v5 (Gated-Only, ARF-10)": CGTv5CalibratedController(tau_gate=0.30, max_rho=0.20, bg_train_step=0, n_heavy_trees=10, seed=42),
            "CGT-v5 (Anti-Staleness Bg=4, ARF-10)": CGTv5CalibratedController(tau_gate=0.30, max_rho=0.20, bg_train_step=4, n_heavy_trees=10, seed=42)
        }

    # Evaluate Stream 1
    print("\n" + "=" * 80)
    print("RUNNING BENCHMARK ON STREAM 1 (Industrial SCADA)")
    print("=" * 80)
    models_s1 = get_models()
    results_s1 = {}
    preds_s1 = {}
    for name, m in models_s1.items():
        print(f"Evaluating {name}...", end=" ", flush=True)
        res = evaluate_stream_detailed(m, test_stream1)
        results_s1[name] = res
        preds_s1[name] = res["y_preds"]
        print(f"-> Acc: {res['accuracy_pct']}%, Mean: {res['mean_cpu_us']} us, Max: {res['max_us']} us, Duty: {res['duty_cycle']*100:.2f}%")

    # Evaluate Stream 2 (POSOCO)
    print("\n" + "=" * 80)
    print("RUNNING BENCHMARK ON STREAM 2 (POSOCO Indian Grid)")
    print("=" * 80)
    models_s2 = get_models()
    results_s2 = {}
    preds_s2 = {}
    for name, m in models_s2.items():
        print(f"Evaluating {name}...", end=" ", flush=True)
        res = evaluate_stream_detailed(m, test_stream2)
        results_s2[name] = res
        preds_s2[name] = res["y_preds"]
        print(f"-> Acc: {res['accuracy_pct']}%, Mean: {res['mean_cpu_us']} us, Max: {res['max_us']} us, Duty: {res['duty_cycle']*100:.2f}%")

    # Assemble Results Master Table
    rows = []
    labels = {
        "Linear (SGD)": "Pre-hoc",
        "Always-HT": "Pre-hoc",
        "Always-ARF (10 Trees)": "Pre-hoc Ref",
        "ARF-3 (Fair Protocol)": "Pre-hoc",
        "CGT-v4 (Amortized, ARF-3)": "Post-hoc v4",
        "CGT-v5 (Gated-Only, ARF-10)": "Post-hoc v5 (Gated)",
        "CGT-v5 (Anti-Staleness Bg=4, ARF-10)": "Post-hoc v5 (Anti-Staleness)"
    }

    for name in models_s1.keys():
        r1 = results_s1[name]
        r2 = results_s2[name]
        rows.append({
            "Model": name,
            "Label": labels[name],
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

    df_master = pd.DataFrame(rows)
    os.makedirs("results/tables", exist_ok=True)
    df_master.to_csv("results/tables/cgt_v5_master_benchmark.csv", index=False)
    print("\nSaved Master Benchmark Table to results/tables/cgt_v5_master_benchmark.csv")
    print(df_master.to_string(index=False))

    # Weekly Block Hypothesis Tests
    print("\n" + "=" * 80)
    print("PAIRED WILCOXON SIGNED-RANK TESTS ON WEEKLY ACCURACY BLOCKS")
    print("=" * 80)
    df_blocks_s1 = compute_weekly_blocks(y_trues_s1, preds_s1, 168)
    df_blocks_s2 = compute_weekly_blocks(y_trues_s2, preds_s2, 168)

    test_rows = []
    comparisons = [
        ("CGT-v5 (Anti-Staleness Bg=4, ARF-10)", "ARF-3 (Fair Protocol)"),
        ("CGT-v5 (Anti-Staleness Bg=4, ARF-10)", "Linear (SGD)"),
        ("CGT-v5 (Anti-Staleness Bg=4, ARF-10)", "CGT-v4 (Amortized, ARF-3)"),
        ("CGT-v5 (Gated-Only, ARF-10)", "ARF-3 (Fair Protocol)"),
    ]

    for stream_idx, (df_b, s_name) in enumerate([(df_blocks_s1, "Stream 1"), (df_blocks_s2, "Stream 2 (POSOCO)")]):
        raw_pvals = []
        stats_list = []
        for m_test, m_base in comparisons:
            diff = df_b[m_test] - df_b[m_base]
            if np.all(diff == 0):
                w_stat, p_val = 0.0, 1.0
            else:
                w_stat, p_val = stats.wilcoxon(df_b[m_test], df_b[m_base], zero_method="pratt")
            stats_list.append((m_test, m_base, w_stat, p_val, float(np.mean(diff))))
            raw_pvals.append(p_val)

        # Holm-Bonferroni correction
        order = np.argsort(raw_pvals)
        m_tests = len(raw_pvals)
        corr_pvals = [0.0] * m_tests
        for rank, idx in enumerate(order):
            corr_pvals[idx] = min(1.0, raw_pvals[idx] * (m_tests - rank))

        for idx, (m_test, m_base, w_stat, p_val, mean_diff) in enumerate(stats_list):
            test_rows.append({
                "Stream": s_name,
                "Comparison": f"{m_test} vs {m_base}",
                "Mean_Weekly_Diff_Pct": round(mean_diff, 3),
                "Wilcoxon_W": round(w_stat, 1),
                "Raw_p_value": f"{p_val:.4e}",
                "Holm_Bonferroni_p": f"{corr_pvals[idx]:.4e}",
                "Significant_0.05": "YES" if corr_pvals[idx] < 0.05 else "NO"
            })

    df_tests = pd.DataFrame(test_rows)
    df_tests.to_csv("results/tables/cgt_v5_weekly_block_tests.csv", index=False)
    print("\nSaved Weekly Block Tests to results/tables/cgt_v5_weekly_block_tests.csv")
    print(df_tests.to_string(index=False))

    # Dual-Stream Pareto Plot
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    for i, (ax, s_col_acc, s_col_lat, s_title) in enumerate([
        (axes[0], "Stream1_Acc_Pct", "Stream1_Mean_us", "Stream 1: Industrial SCADA (24,528h)"),
        (axes[1], "Stream2_POSOCO_Acc_Pct", "Stream2_POSOCO_Mean_us", "Stream 2: Real POSOCO Indian Grid (16,800h)")
    ]):
        for _, r in df_master.iterrows():
            m_name = r["Model"]
            acc = r[s_col_acc]
            lat = r[s_col_lat]

            if "CGT-v5" in m_name:
                color = "#00c853"
                marker = "*"
                size = 180
            elif "CGT-v4" in m_name:
                color = "#2962ff"
                marker = "D"
                size = 120
            elif "10 Trees" in m_name:
                color = "#d50000"
                marker = "X"
                size = 130
            elif "ARF-3" in m_name:
                color = "#aa00ff"
                marker = "s"
                size = 110
            else:
                color = "#757575"
                marker = "o"
                size = 90

            ax.scatter(lat, acc, c=color, marker=marker, s=size, edgecolors="black", linewidths=1.2, zorder=4)
            offset_x = 1.08 if lat < 1000 else 0.65
            ax.annotate(m_name.replace(" (Post-hoc)", ""), (lat * offset_x, acc - 0.25), fontsize=8, fontweight="bold")

        ax.axvline(1000, color="red", linestyle="--", alpha=0.7, label="1,000 µs Sub-1ms Budget Limit")
        ax.set_xscale("log")
        ax.set_xlabel("Mean Inference Latency (µs, Log Scale)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Prequential Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title(s_title, fontsize=12, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.suptitle("CGT-v5 Calibrated Epistemic Cascade: Dual-Stream Pareto Dominance\n(Post-hoc: Developed After Viewing Test Results)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = "results/figures/fig_cgt_v5_accuracy_vs_latency_both_streams.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved publication plot to {fig_path}")


if __name__ == "__main__":
    main()
