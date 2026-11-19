#!/usr/bin/env python3
"""
scripts/audit_sgd_vs_cgt_tail.py
================================
Standalone read-only comparative tail-risk and stability audit between:
1. Streaming Linear (SGD)
2. CGT-v4 (Amortized, ARF-3) [Post-hoc: developed after viewing test results]

Evaluated across both continuous streams:
- Stream 1: Primary SCADA Benchmark Stream (24,528 test hours, 146 weekly blocks)
- Stream 2: Real POSOCO Indian National Grid Stream (16,800 test hours, 100 weekly blocks)

Metrics Computed:
(1) Overall Accuracy (%)
(2) Worst single week accuracy (Min Week %)
(3) Mean of the worst 10% of weeks (%)
(4) Weekly standard deviation (%)
(5) Maximum consecutive error streak (hours)

Strictly read-only on model and data structures.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add repository root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.stream_loader import ElectricityStreamLoader
from scripts.run_real_posoco_benchmark import load_real_posoco_stream
from scripts.run_cgt_v4_master_benchmark import LinearSGDWrapper
from src.controllers.cgt_v4 import CGTv4Controller


def compute_tail_metrics(y_true, y_pred, block_size=168):
    """
    Computes summary and tail-risk distribution metrics over prequential predictions.
    """
    n = len(y_true)
    n_blocks = n // block_size
    weekly_accs = []
    
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        acc = np.mean(y_pred[start:end] == y_true[start:end]) * 100.0
        weekly_accs.append(acc)
        
    weekly_accs = np.array(weekly_accs)
    
    # 1. Overall Accuracy
    overall_acc = np.mean(y_pred == y_true) * 100.0
    
    # 2. Worst Single Week Accuracy (Min Week)
    min_week = np.min(weekly_accs)
    
    # 3. Mean of the Worst 10% of Weeks
    k_10pct = max(1, int(np.ceil(0.10 * len(weekly_accs))))
    worst_10pct_mean = np.mean(np.sort(weekly_accs)[:k_10pct])
    
    # 4. Weekly Standard Deviation
    weekly_std = np.std(weekly_accs, ddof=1)
    
    # 5. Maximum Consecutive Error Streak
    errors = (y_pred != y_true)
    max_streak = 0
    current_streak = 0
    for err in errors:
        if err:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
            
    return {
        "overall_acc": overall_acc,
        "min_week": min_week,
        "worst_10pct_mean": worst_10pct_mean,
        "weekly_std": weekly_std,
        "max_streak": max_streak,
        "weekly_accs": weekly_accs
    }


def run_evaluation():
    print("=" * 85)
    print("AUDIT: STREAMING LINEAR (SGD) VS. CGT-v4 (AMORTIZED)")
    print("Tail-Risk, Weekly Volatility, and Maximum Error Streak Assessment")
    print("=" * 85)
    
    # Load frozen configuration
    config_path = os.path.join(BASE_DIR, "configs", "cgt_v4_frozen_config.json")
    with open(config_path, "r") as f:
        frozen_data = json.load(f)
    cfg_amortized = frozen_data["cgt_v4_amortized_arf3"]
    print(f"Loaded frozen CGT-v4 config (SHA-256: {frozen_data['metadata']['sha256'][:16]}...)")
    
    records = []
    
    # -------------------------------------------------------------------------
    # PART 1: STREAM 1 (PRIMARY SCADA BENCHMARK, 24,528 TEST HOURS)
    # -------------------------------------------------------------------------
    print("\n[1/2] Processing Stream 1 (Primary SCADA Benchmark: 24,528 test hours, 146 weeks)...")
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    full_stream1 = list(loader.iter_stream(warmup_rows=24))
    val_len1 = 10512
    test_len1 = 24528
    test_stream1 = full_stream1[val_len1:val_len1 + test_len1]
    y_true_s1 = np.array([item[1] for item in test_stream1])
    
    # Evaluate Linear (SGD)
    m_lin1 = LinearSGDWrapper()
    p_lin1 = []
    for item in test_stream1:
        p = m_lin1.predict_one(item[0])
        m_lin1.learn_one(item[0], item[1])
        p_lin1.append(p)
    p_lin1 = np.array(p_lin1)
    res_lin1 = compute_tail_metrics(y_true_s1, p_lin1, block_size=168)
    
    records.append({
        "Stream": "Stream 1 (Benchmark 24k)",
        "Model": "Linear (SGD)",
        "Overall_Accuracy_Pct": round(res_lin1["overall_acc"], 3),
        "Min_Week_Accuracy_Pct": round(res_lin1["min_week"], 2),
        "Worst_10Pct_Weeks_Mean_Pct": round(res_lin1["worst_10pct_mean"], 2),
        "Weekly_Std_Pct": round(res_lin1["weekly_std"], 2),
        "Max_Consecutive_Error_Streak": res_lin1["max_streak"]
    })
    
    # Evaluate CGT-v4 (Amortized, ARF-3)
    m_cgt1 = CGTv4Controller(
        tau_gate=cfg_amortized["tau_gate"],
        max_rho=cfg_amortized["max_rho"],
        h_cooldown=cfg_amortized["h_cooldown"],
        replay_mode=cfg_amortized["replay_mode"],
        buffer_b=cfg_amortized["buffer_b"],
        k_replay_per_step=cfg_amortized["k_replay_per_step"],
        batch_size=cfg_amortized["batch_size"],
        n_heavy_trees=3,
        seed=42
    )
    p_cgt1 = []
    for item in test_stream1:
        p = m_cgt1.predict_one(item[0])
        m_cgt1.learn_one(item[0], item[1], p)
        p_cgt1.append(p)
    p_cgt1 = np.array(p_cgt1)
    res_cgt1 = compute_tail_metrics(y_true_s1, p_cgt1, block_size=168)
    
    records.append({
        "Stream": "Stream 1 (Benchmark 24k)",
        "Model": "CGT-v4 (Amortized, ARF-3)",
        "Overall_Accuracy_Pct": round(res_cgt1["overall_acc"], 3),
        "Min_Week_Accuracy_Pct": round(res_cgt1["min_week"], 2),
        "Worst_10Pct_Weeks_Mean_Pct": round(res_cgt1["worst_10pct_mean"], 2),
        "Weekly_Std_Pct": round(res_cgt1["weekly_std"], 2),
        "Max_Consecutive_Error_Streak": res_cgt1["max_streak"]
    })
    
    # -------------------------------------------------------------------------
    # PART 2: STREAM 2 (REAL POSOCO INDIAN GRID, 16,800 TEST HOURS)
    # -------------------------------------------------------------------------
    print("[2/2] Processing Stream 2 (POSOCO Indian National Grid: 16,800 test hours, 100 weeks)...")
    full_stream2 = load_real_posoco_stream("data/india_electricity_real_hourly.csv", warmup_rows=24)
    val_len2 = 3624
    test_len2 = 16800
    test_stream2 = full_stream2[val_len2:val_len2 + test_len2]
    y_true_s2 = np.array([item[1] for item in test_stream2])
    
    # Evaluate Linear (SGD)
    m_lin2 = LinearSGDWrapper()
    p_lin2 = []
    for item in test_stream2:
        p = m_lin2.predict_one(item[0])
        m_lin2.learn_one(item[0], item[1])
        p_lin2.append(p)
    p_lin2 = np.array(p_lin2)
    res_lin2 = compute_tail_metrics(y_true_s2, p_lin2, block_size=168)
    
    records.append({
        "Stream": "Stream 2 (POSOCO 16.8k)",
        "Model": "Linear (SGD)",
        "Overall_Accuracy_Pct": round(res_lin2["overall_acc"], 3),
        "Min_Week_Accuracy_Pct": round(res_lin2["min_week"], 2),
        "Worst_10Pct_Weeks_Mean_Pct": round(res_lin2["worst_10pct_mean"], 2),
        "Weekly_Std_Pct": round(res_lin2["weekly_std"], 2),
        "Max_Consecutive_Error_Streak": res_lin2["max_streak"]
    })
    
    # Evaluate CGT-v4 (Amortized, ARF-3)
    m_cgt2 = CGTv4Controller(
        tau_gate=cfg_amortized["tau_gate"],
        max_rho=cfg_amortized["max_rho"],
        h_cooldown=cfg_amortized["h_cooldown"],
        replay_mode=cfg_amortized["replay_mode"],
        buffer_b=cfg_amortized["buffer_b"],
        k_replay_per_step=cfg_amortized["k_replay_per_step"],
        batch_size=cfg_amortized["batch_size"],
        n_heavy_trees=3,
        seed=42
    )
    p_cgt2 = []
    for item in test_stream2:
        p = m_cgt2.predict_one(item[0])
        m_cgt2.learn_one(item[0], item[1], p)
        p_cgt2.append(p)
    p_cgt2 = np.array(p_cgt2)
    res_cgt2 = compute_tail_metrics(y_true_s2, p_cgt2, block_size=168)
    
    records.append({
        "Stream": "Stream 2 (POSOCO 16.8k)",
        "Model": "CGT-v4 (Amortized, ARF-3)",
        "Overall_Accuracy_Pct": round(res_cgt2["overall_acc"], 3),
        "Min_Week_Accuracy_Pct": round(res_cgt2["min_week"], 2),
        "Worst_10Pct_Weeks_Mean_Pct": round(res_cgt2["worst_10pct_mean"], 2),
        "Weekly_Std_Pct": round(res_cgt2["weekly_std"], 2),
        "Max_Consecutive_Error_Streak": res_cgt2["max_streak"]
    })
    
    df_results = pd.DataFrame(records)
    
    print("\n" + "=" * 85)
    print("TAIL-RISK & STABILITY AUDIT: LINEAR (SGD) VS. CGT-v4 (AMORTIZED)")
    print("=" * 85)
    # Format clean Markdown table
    header = "| Stream | Model | Overall Acc (%) | Min Week Acc (%) | Worst 10% Weeks Mean (%) | Weekly Std (%) | Max Error Streak (h) |"
    divider = "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    rows = []
    for r in records:
        rows.append(f"| {r['Stream']} | {r['Model']} | {r['Overall_Accuracy_Pct']:.3f}% | {r['Min_Week_Accuracy_Pct']:.2f}% | {r['Worst_10Pct_Weeks_Mean_Pct']:.2f}% | {r['Weekly_Std_Pct']:.2f}% | {r['Max_Consecutive_Error_Streak']} |")
    
    table_str = "\n".join([header, divider] + rows)
    print(table_str)
    print("=" * 85)
    
    # Save results to results/tables/audit_sgd_vs_cgt_tail.csv
    out_csv = os.path.join(BASE_DIR, "results", "tables", "audit_sgd_vs_cgt_tail.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved audit summary to: {out_csv}")
    
    return df_results


if __name__ == "__main__":
    run_evaluation()
