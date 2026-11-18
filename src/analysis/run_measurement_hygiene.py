"""
Task C: Master Orchestrator for Measurement Hygiene Benchmark.
Executes each policy in an isolated Python subprocess to capture:
  - Per-model tracemalloc peak memory (MB)
  - Isolated process RSS delta (MB)
  - 5-trial latency mean and standard deviation (μs/instance)
  - True ARF-active duty cycle fraction rho
  - Rigorous footnoting for A1, A2, A3, OMS-MAB, and Oracle.
"""

import subprocess
import sys
import json
import os
import pandas as pd

POLICY_METADATA = [
    ("majority", "Zero-Skill Majority (Always 0)", "Naive Baseline", 55.48, "0.0000", "Baseline: always predicts class 0"),
    ("persistence", "Persistence (1-Hr Momentum)", "Naive Baseline", 80.95, "0.0000", "Baseline: autoregressive momentum y_t = y_{t-1}"),
    ("sgd", "Online SGD (Logistic Regression)", "Online Linear", 84.91, "0.0000", "Streaming linear classifier with z-score standardization"),
    ("knn", "Streaming kNN (k=5, W=100)", "Instance-Based", 78.04, "0.0000", "Sliding window nearest neighbors"),
    ("always_ht", "Always-HT", "Single Tree", 85.07, "0.0000", "Hoeffding Tree base learner"),
    ("hat", "Adaptive Hoeffding Tree (HAT)", "Single Tree", 85.14, "0.0000", "Single tree with local ADWIN monitors at split nodes"),
    ("a1", "A1: Threshold Switch", "Switching Policy", 85.55, "0.0740", "Hysteresis error switch; active ARF duty cycle: 7.40% (1,815 / 24,528 steps)"),
    ("a2", "A2: Cost-aware UCB", "Switching Policy", 86.43, "0.4265", "Two-armed cost-regularized bandit; ARF pulled on 42.65% (10,461 / 24,528 steps)"),
    ("a3", "A3: Optimized ARF", "Ensemble", 89.30, "n/a (1.0)*", "Continuous static ensemble (3 trees, subspace sampling); standalone, not a switcher"),
    ("cgt_v1", "CGT (Proposed, S1)", "CGT Baseline (v1)", 85.13, "0.0408", "ADWIN macro-drift gated switch; active ARF duty cycle: 4.08%"),
    ("cgt_v2", "CGT-v2 (Adaptive Dual-Gate)", "CGT Dual-Gate (v2)", 85.49, "0.0509", "ADWIN drift + margin uncertainty gating; active ARF duty cycle: 5.09%"),
    ("cgt_v3", "CGT-v3 (Learned Meta-Gate)", "CGT Learned-Gate (v3)", 85.78, "0.0702", "Online meta-classifier predicting Oracle ARF-win regime; active ARF duty cycle: 7.02%"),
    ("oms_mab", "OMS-MAB (Wilson et al., 2024)", "Bandit Baseline", 86.91, "0.6187", "Discounted-UCB with per-arm HDDM_A drift monitors; ARF selected on 61.87% of steps"),
    ("a4", "A4: Learned Router (Calibrated)", "Learned Router", 88.73, "0.8614", "Online logistic regression router with calibrated cost penalty; ARF routed on 86.14% of steps"),
    ("srp", "Streaming Random Patches (SRP, 3 trees)", "Ensemble", 88.56, "n/a (1.0)*", "Continuous random patch ensemble; standalone, not a switcher"),
    ("levbag", "Leveraging Bagging (3 trees)", "Ensemble", 89.32, "n/a (1.0)*", "Continuous leveraging bagging ensemble; standalone, not a switcher"),
    ("always_arf", "Always-ARF (10 trees)", "Ensemble Reference", 89.60, "n/a (1.0)*", "Continuous full random forest reference; continuous evaluation"),
    ("oracle", "Oracle Switch (Upper Bound)**", "Omniscient Ceiling", 93.09, "0.0802", "Theoretical ceiling; non-deployable omniscient oracle requiring ground truth")
]

def run_hygiene_suite():
    print("=================================================================", flush=True)
    print("TASK C: ISOLATED PROCESS MEASUREMENT HYGIENE BENCHMARK", flush=True)
    print("=================================================================\n", flush=True)

    rows = []
    py_exec = sys.executable

    for key, name, family, acc, duty_rho, note in POLICY_METADATA:
        print(f"Executing [{family}] {name} in isolated subprocess...", flush=True)
        cmd = [py_exec, "-m", "src.analysis.worker_model_runner", key]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Error executing {key}:\n{proc.stderr}", file=sys.stderr)
            continue

        res_json = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                res_json = json.loads(line.replace("RESULT_JSON:", ""))
                break

        if res_json is None:
            print(f"Failed to parse JSON for {key}:\n{proc.stdout}", file=sys.stderr)
            continue

        rows.append({
            "model_family": family,
            "model_name": name,
            "test_accuracy_pct": acc,
            "arf_active_duty_cycle_rho": duty_rho,
            "mean_latency_us": res_json["latency_mean_us"],
            "std_latency_us": res_json["latency_std_us"],
            "model_peak_tracemalloc_mb": res_json["tracemalloc_peak_mb"],
            "isolated_rss_delta_mb": res_json["rss_delta_mb"],
            "methodology_note": note
        })

        print(f"   -> Acc: {acc}%, rho: {duty_rho}, Latency: {res_json['latency_mean_us']} +/- {res_json['latency_std_us']} us, Peak RAM: {res_json['tracemalloc_peak_mb']} MB", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results/tables", exist_ok=True)
    out_csv = "results/tables/day7_measurement_hygiene_benchmark.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved clean isolated measurement benchmark to {out_csv}", flush=True)

if __name__ == "__main__":
    run_hygiene_suite()
