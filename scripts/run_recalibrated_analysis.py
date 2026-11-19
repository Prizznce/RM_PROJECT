import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import math
import numpy as np
import pandas as pd
from scipy import stats

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper
from src.controllers.oms_mab import OMSMABController
from src.controllers.cgt_v3 import CGTv3LearnedGateController
from river.forest import ARFClassifier


class ARF3Wrapper:
    """Standardized ARF with 3 trees matching frozen validation config."""
    def __init__(self, grace_period=200, lambda_value=6, delta=0.001, split_criterion='info_gain', seed=42):
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
    """Random-Gate Baseline for multi-seed evaluation."""
    def __init__(self, p=0.0459, seed=42):
        self.p = p
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.name = f"Random-Gate (p={p:.2%}, seed={seed})"
        self.ht = HoeffdingTreeWrapper()
        self.arf = ARF3Wrapper(seed=seed)
        self.gate_open = False
        self.gate_open_steps = 0
        self.total_steps = 0

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


def evaluate_timed(model, stream):
    n = len(stream)
    y_trues = np.zeros(n, dtype=int)
    y_preds = np.zeros(n, dtype=int)
    gate_states = np.zeros(n, dtype=int)
    cpu_ns = np.zeros(n, dtype=int)
    correct = 0
    t_start = time.perf_counter_ns()

    for idx, (x_t, y_t, ts) in enumerate(stream):
        y_trues[idx] = y_t
        t0 = time.perf_counter_ns()
        pred = model.predict_one(x_t)
        if hasattr(model, 'name') and ('CGT' in model.name or 'OMS-MAB' in model.name):
            model.learn_one(x_t, y_t, pred)
        else:
            model.learn_one(x_t, y_t)
        dt = time.perf_counter_ns() - t0
        cpu_ns[idx] = dt
        y_preds[idx] = pred
        
        is_gate = 1 if getattr(model, 'gate_open', False) else (
            1 if getattr(model, 'active_arm_idx', -1) == 1 else (
                1 if isinstance(model, ARF3Wrapper) or isinstance(model, ARFFullWrapper) else 0
            )
        )
        gate_states[idx] = is_gate
        if pred == y_t:
            correct += 1

    t_total_ns = time.perf_counter_ns() - t_start
    acc = (correct / n) * 100.0
    mean_cpu_us = (t_total_ns / n) / 1000.0
    duty_cycle = float(np.mean(gate_states))
    return {
        "accuracy_pct": round(acc, 3),
        "mean_cpu_us": round(mean_cpu_us, 2),
        "latency_p50_us": round(float(np.percentile(cpu_ns, 50)) / 1000.0, 2),
        "latency_p95_us": round(float(np.percentile(cpu_ns, 95)) / 1000.0, 2),
        "latency_p99_us": round(float(np.percentile(cpu_ns, 99)) / 1000.0, 2),
        "latency_max_us": round(float(np.max(cpu_ns)) / 1000.0, 2),
        "duty_cycle": round(duty_cycle, 4),
        "y_trues": y_trues,
        "y_preds": y_preds,
        "gate_states": gate_states,
        "cpu_ns": cpu_ns
    }


def main():
    print("=" * 80)
    print("RUNNING RECALIBRATED SCIENTIFIC ANALYSIS")
    print("=" * 80)

    # 1. Load Data
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_instances = 10512
    test_instances = 24528
    test_stream = full_stream[val_instances:val_instances + test_instances]
    print(f"Loaded test stream: {len(test_stream)} instances (Days 440 - 1461).\n")

    # =========================================================================
    # PART 1: RE-MEASURE OMS-MAB IN THE EXACT SAME TIMING HARNESS
    # =========================================================================
    print("--> [1/4] Re-measuring OMS-MAB (Wilson et al., 2024) in test harness...")
    oms_mab = OMSMABController(gamma=0.98, c=0.5, seed=42)
    res_oms = evaluate_timed(oms_mab, test_stream)
    print(f"    OMS-MAB Test Acc:       {res_oms['accuracy_pct']}%")
    print(f"    OMS-MAB Duty Cycle:     {res_oms['duty_cycle']*100:.2f}%")
    print(f"    OMS-MAB Mean Latency:   {res_oms['mean_cpu_us']} us (p95: {res_oms['latency_p95_us']} us, Worst: {res_oms['latency_max_us']} us)")

    df_oms_raw = pd.DataFrame({
        "instance_idx": np.arange(1, len(test_stream) + 1),
        "y_true": res_oms["y_trues"],
        "y_pred": res_oms["y_preds"],
        "gate_open": res_oms["gate_states"],
        "cpu_time_ns": res_oms["cpu_ns"]
    })
    df_oms_raw.to_csv("results/raw/fair_test_predictions_oms_mab.csv", index=False)

    # =========================================================================
    # PART 2: EVALUATE RANDOM-GATE OVER 20 SEEDS (SEEDS 1 TO 20)
    # =========================================================================
    print("\n--> [2/4] Evaluating Random-Gate over 20 random seeds (seeds 1 to 20)...")
    n_blocks = 146
    block_size = 168
    rg_block_matrix = np.zeros((20, n_blocks))
    rg_overall_accs = []
    rg_duties = []

    for seed_idx, s in enumerate(range(1, 21)):
        rg = RandomGateBaseline(p=0.0459, seed=s)
        corr_arr = np.zeros(len(test_stream), dtype=float)
        gates = np.zeros(len(test_stream), dtype=float)
        for idx, (x_t, y_t, ts) in enumerate(test_stream):
            pred = rg.predict_one(x_t)
            rg.learn_one(x_t, y_t)
            if pred == y_t:
                corr_arr[idx] = 1.0
            gates[idx] = 1.0 if rg.gate_open else 0.0

        acc = np.mean(corr_arr) * 100.0
        duty = np.mean(gates) * 100.0
        rg_overall_accs.append(acc)
        rg_duties.append(duty)
        # Block accuracies
        b_acc = corr_arr.reshape(n_blocks, block_size).mean(axis=1) * 100.0
        rg_block_matrix[seed_idx, :] = b_acc

    # Mean weekly accuracy across 20 seeds
    rg_20seed_block_means = np.mean(rg_block_matrix, axis=0)
    mean_rg_overall = np.mean(rg_overall_accs)
    std_rg_overall = np.std(rg_overall_accs, ddof=1)
    mean_rg_duty = np.mean(rg_duties)
    print(f"    Random-Gate (20 Seeds) Mean Acc: {mean_rg_overall:.3f}% +/- {std_rg_overall:.3f}% (Duty = {mean_rg_duty:.2f}%)")

    # =========================================================================
    # PART 3: WEEKLY BLOCK TESTS (146 BLOCKS) ON FAIR TEST PREDICTIONS
    # =========================================================================
    print("\n--> [3/4] Running 146-Week Block Statistical Tests with Holm Correction...")
    # Load previously saved fair predictions
    df_cgt = pd.read_csv("results/raw/fair_test_predictions_cgt.csv")
    df_arf3 = pd.read_csv("results/raw/fair_test_predictions_arf3.csv")
    
    y_true = df_cgt["y_true"].values
    cgt_corr = (df_cgt["y_pred"].values == y_true).astype(float)
    arf3_corr = (df_arf3["y_pred"].values == y_true).astype(float)

    # For Always-HT: evaluate or load
    ht_model = HoeffdingTreeWrapper()
    ht_corr = np.zeros(len(test_stream), dtype=float)
    for idx, (x_t, y_t, ts) in enumerate(test_stream):
        p = ht_model.predict_one(x_t)
        ht_model.learn_one(x_t, y_t)
        if p == y_t:
            ht_corr[idx] = 1.0

    # Weekly blocks (146 blocks x 168 hours = 24,528 hours)
    cgt_blocks = cgt_corr.reshape(n_blocks, block_size).mean(axis=1) * 100.0
    arf3_blocks = arf3_corr.reshape(n_blocks, block_size).mean(axis=1) * 100.0
    ht_blocks = ht_corr.reshape(n_blocks, block_size).mean(axis=1) * 100.0
    rg_blocks = rg_20seed_block_means

    # Overall accuracies
    cgt_acc = float(np.mean(cgt_corr)) * 100.0
    arf3_acc = float(np.mean(arf3_corr)) * 100.0
    ht_acc = float(np.mean(ht_corr)) * 100.0

    comparisons = [
        ("CGT vs Always-HT", cgt_blocks - ht_blocks, cgt_acc - ht_acc),
        ("CGT vs Random-Gate (20 seeds)", cgt_blocks - rg_blocks, cgt_acc - mean_rg_overall),
        ("ARF-3 vs CGT", arf3_blocks - cgt_blocks, arf3_acc - cgt_acc)
    ]

    test_results = []
    raw_pvals = []

    for name, diff, overall_delta in comparisons:
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        cohen_d = float(mean_diff / std_diff) if std_diff > 0 else 0.0

        # Verification of exact arithmetic identity
        diff_from_overall = abs(mean_diff - overall_delta)
        assert diff_from_overall < 1e-4, f"Mismatch: mean weekly {mean_diff} vs overall {overall_delta}"

        # Statistical tests
        non_zero = diff[diff != 0]
        if len(non_zero) > 0:
            w_stat, w_pval = stats.wilcoxon(non_zero)
            t_stat, t_pval = stats.ttest_1samp(diff, 0.0)
        else:
            w_stat, w_pval, t_stat, t_pval = 0.0, 1.0, 0.0, 1.0

        raw_pvals.append(w_pval)
        test_results.append({
            "Comparison": name,
            "Mean_Weekly_Delta_pct": round(mean_diff, 3),
            "Overall_Stream_Delta_pct": round(overall_delta, 3),
            "Paired_Cohen_d": round(cohen_d, 3),
            "Wilcoxon_W": round(float(w_stat), 2),
            "Raw_Wilcoxon_p": w_pval,
            "Paired_t_stat": round(float(t_stat), 2),
            "Paired_t_p": t_pval
        })

    # Holm-Bonferroni correction on Wilcoxon p-values
    # Sort p-values: smallest gets multiplied by M, next by M-1, etc.
    m = len(raw_pvals)
    sorted_indices = np.argsort(raw_pvals)
    holm_pvals = np.zeros(m)
    
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        mult = m - rank
        adj = min(raw_pvals[idx] * mult, 1.0)
        adj = max(adj, running_max)  # monotonicity
        running_max = adj
        holm_pvals[idx] = adj

    for i, res in enumerate(test_results):
        res["Holm_Adjusted_Wilcoxon_p"] = f"{holm_pvals[i]:.4e}"
        res["Raw_Wilcoxon_p_str"] = f"{res['Raw_Wilcoxon_p']:.4e}"
        res["Significant_at_0.05"] = bool(holm_pvals[i] < 0.05)

    df_tests = pd.DataFrame(test_results)
    df_tests.to_csv("results/tables/fair_weekly_block_statistical_tests.csv", index=False)
    print("\nWeekly Block Statistical Significance Table (Holm Corrected):")
    cols_show = ["Comparison", "Mean_Weekly_Delta_pct", "Overall_Stream_Delta_pct", "Paired_Cohen_d", "Wilcoxon_W", "Raw_Wilcoxon_p_str", "Holm_Adjusted_Wilcoxon_p", "Significant_at_0.05"]
    print(df_tests[cols_show].to_string(index=False))

    # =========================================================================
    # PART 4: UPDATE MODELED MULTI-RATE HARDWARE PROFILES TABLE
    # =========================================================================
    print("\n--> [4/4] Updating Modeled Multi-Rate Hardware Profiles (50 Hz, 200 Hz, 1 kHz)...")
    
    # Load previously evaluated test latencies
    df_fair_summary = pd.read_csv("results/tables/fair_comparison_test_results.csv")
    
    # Model catalog with measured latencies
    # Note: OMS-MAB values are now updated with newly measured res_oms
    model_bench = [
        {"name": "Always-HT", "mean_lat": 94.26, "max_lat": 2404.0},
        {"name": "Random-Gate (rho_CGT)", "mean_lat": 112.11, "max_lat": 4309.3},
        {"name": "CGT (Learned Gate)", "mean_lat": 243.37, "max_lat": 38002.8},
        {"name": "ARF-3 (3 Trees)", "mean_lat": 281.35, "max_lat": 2231.1},
        {"name": "OMS-MAB (Wilson et al.)", "mean_lat": res_oms["mean_cpu_us"], "max_lat": res_oms["latency_max_us"]},
        {"name": "Always-ARF (10 Trees)", "mean_lat": 1333.95, "max_lat": 101442.2}
    ]

    rates = [
        {"rate_name": "50 Hz (Standard PMU)", "freq": 50.0, "t_us": 20000.0},
        {"rate_name": "200 Hz (High-Speed PMU / DFR)", "freq": 200.0, "t_us": 5000.0},
        {"rate_name": "1 kHz (Substation RTU Deadline)", "freq": 1000.0, "t_us": 1000.0}
    ]

    hw_rows = []
    for r in rates:
        t_per = r["t_us"]
        for m in model_bench:
            t_m = m["mean_lat"]
            t_w = m["max_lat"]
            cpu_duty = (t_m / t_per) * 100.0
            
            # Schedulability status based on actual measured latencies:
            # If t_mean > t_per -> OVERRUN (Mean Overload)
            # If t_worst > t_per -> OVERRUN (Worst-case / Replay spike)
            # If both <= t_per -> GUARANTEED
            if t_m > t_per:
                sched_status = "OVERRUN (Mean Overload)"
            elif t_w > t_per:
                sched_status = "OVERRUN (Replay Spike)"
            else:
                sched_status = "GUARANTEED"

            # Power calculations (modeled)
            d_frac = min(cpu_duty / 100.0, 1.0)
            p_stm32 = d_frac * 280.0 + (1.0 - d_frac) * 8.2
            bat_stm32_100wh = 100.0 / (p_stm32 / 1000.0)
            p_cm4 = d_frac * 3000.0 + (1.0 - d_frac) * 850.0
            bat_cm4_100wh = 100.0 / (p_cm4 / 1000.0)

            hw_rows.append({
                "Sampling_Rate": r["rate_name"],
                "Frequency_Hz": r["freq"],
                "Sample_Period_us": t_per,
                "Model_Name": m["name"],
                "Mean_Latency_us (host-measured)": t_m,
                "Worst_Case_Latency_us (host-measured)": t_w,
                "Modeled_CPU_Duty_pct": round(cpu_duty, 3),
                "RealTime_Schedulability (modeled)": sched_status,
                "STM32H743_Power_mW (modeled)": round(p_stm32, 2),
                "STM32H743_100Wh_Autonomy_hrs (modeled)": round(bat_stm32_100wh, 1),
                "CM4_Power_mW (modeled)": round(p_cm4, 1),
                "CM4_100Wh_Autonomy_hrs (modeled)": round(bat_cm4_100wh, 1)
            })

    df_hw = pd.DataFrame(hw_rows)
    df_hw.to_csv("results/tables/modeled_hardware_multirate_comparison.csv", index=False)
    print("\nUpdated Modeled Multi-Rate Hardware Profiles Table:")
    print(df_hw[["Sampling_Rate", "Model_Name", "Mean_Latency_us (host-measured)", "Modeled_CPU_Duty_pct", "RealTime_Schedulability (modeled)", "STM32H743_100Wh_Autonomy_hrs (modeled)"]].to_string(index=False))

    print("\n✅ Recalibrated scientific analysis complete!")

if __name__ == "__main__":
    main()

