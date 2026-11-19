import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import math
import tempfile
import numpy as np
import pandas as pd
from scipy import stats

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import matplotlib.pyplot as plt

from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier
from river.drift import ADWIN

from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper
from src.controllers.cgt_v3 import CGTv3LearnedGateController
from src.controllers.oms_mab import OMSMABController
from src.models.stronger_baselines import ZeroSkillMajorityClass, NaivePersistenceHourly


def stream_india_grid(csv_path="data/india_electricity_hourly.csv", warmup_rows=24):
    df = pd.read_csv(csv_path)
    timestamps = pd.to_datetime(df['Timestamp']).values
    national_demands = df['National_Demand_MW'].values
    solars = df['Solar_Gen_MW'].values
    winds = df['Wind_Gen_MW'].values
    coals = df['Thermal_Coal_Gen_MW'].values
    hydros = df['Hydro_Gen_MW'].values
    freqs = df['Grid_Frequency_Hz'].values
    ren_ratios = df['Renewable_Ratio'].values
    nr_demands = df['Northern_Region_Demand_MW'].values
    wr_demands = df['Western_Region_Demand_MW'].values
    sr_demands = df['Southern_Region_Demand_MW'].values
    er_demands = df['Eastern_Region_Demand_MW'].values
    ner_demands = df['NorthEastern_Region_Demand_MW'].values
    targets = df['Target_Stress_Event'].values

    n = len(df)
    for t in range(warmup_rows, n):
        ts = pd.Timestamp(timestamps[t])
        hour = ts.hour
        dow = ts.dayofweek
        month = ts.month

        d_1 = float(national_demands[t - 1])
        d_2 = float(national_demands[t - 2])
        d_24 = float(national_demands[t - 24])

        x_t = {
            'demand_lag_1': d_1,
            'demand_lag_2': d_2,
            'demand_lag_24': d_24,
            'demand_diff_1_2': d_1 - d_2,
            'demand_diff_1_24': d_1 - d_24,
            'solar_lag_1': float(solars[t - 1]),
            'wind_lag_1': float(winds[t - 1]),
            'coal_lag_1': float(coals[t - 1]),
            'hydro_lag_1': float(hydros[t - 1]),
            'freq_lag_1': float(freqs[t - 1]),
            'renewable_ratio_lag_1': float(ren_ratios[t - 1]),
            'nr_demand_lag_1': float(nr_demands[t - 1]),
            'wr_demand_lag_1': float(wr_demands[t - 1]),
            'sr_demand_lag_1': float(sr_demands[t - 1]),
            'er_demand_lag_1': float(er_demands[t - 1]),
            'ner_demand_lag_1': float(ner_demands[t - 1]),
            'hour_sin': float(math.sin(2 * math.pi * hour / 24.0)),
            'hour_cos': float(math.cos(2 * math.pi * hour / 24.0)),
            'day_of_week': int(dow),
            'month': int(month)
        }
        y_t = int(targets[t])
        yield x_t, y_t, ts


class ARF3Wrapper:
    """ARF-3 calibrated for Indian Grid stream."""
    def __init__(self, grace_period=200, lambda_value=6, delta=0.001, seed=42):
        self.name = "ARF-3 (3 Trees)"
        self.model = ARFClassifier(
            n_models=3,
            grace_period=grace_period,
            lambda_value=lambda_value,
            delta=delta,
            split_criterion='info_gain',
            seed=seed
        )

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y, y_pred=None):
        self.model.learn_one(x, y)


class RandomGateBaseline:
    """Random-Gate baseline for Indian Grid stream."""
    def __init__(self, p=0.0571, seed=42):
        self.p = p
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.name = f"Random-Gate (p={p:.2%})"
        self.ht = HoeffdingTreeWrapper(grace_period=200)
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


def evaluate_stream(model, stream, is_timed=True):
    n = len(stream)
    y_trues = np.zeros(n, dtype=int)
    y_preds = np.zeros(n, dtype=int)
    gate_states = np.zeros(n, dtype=int)
    cpu_ns = np.zeros(n, dtype=int) if is_timed else None
    correct = 0
    t_start = time.perf_counter_ns()

    for idx, (x_t, y_t, ts) in enumerate(stream):
        y_trues[idx] = y_t
        if is_timed:
            t0 = time.perf_counter_ns()
            pred = model.predict_one(x_t)
            if hasattr(model, 'name') and ('CGT' in model.name or 'OMS-MAB' in model.name):
                model.learn_one(x_t, y_t, pred)
            else:
                model.learn_one(x_t, y_t)
            dt = time.perf_counter_ns() - t0
            cpu_ns[idx] = dt
        else:
            pred = model.predict_one(x_t)
            if hasattr(model, 'name') and ('CGT' in model.name or 'OMS-MAB' in model.name):
                model.learn_one(x_t, y_t, pred)
            else:
                model.learn_one(x_t, y_t)

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
    
    res = {
        "accuracy_pct": round(acc, 3),
        "mean_cpu_us": round(mean_cpu_us, 2),
        "duty_cycle": round(duty_cycle, 4),
        "y_trues": y_trues,
        "y_preds": y_preds,
        "gate_states": gate_states
    }
    if is_timed:
        res["cpu_ns"] = cpu_ns
        res["latency_p95_us"] = round(float(np.percentile(cpu_ns, 95)) / 1000.0, 2)
        res["latency_max_us"] = round(float(np.max(cpu_ns)) / 1000.0, 2)
    return res


def main():
    print("=" * 80)
    print("100% INDIAN NATIONAL GRID PIPELINE (POSOCO / NLDC 2021-2023)")
    print("=" * 80)

    stream_list = list(stream_india_grid())
    print(f"Loaded {len(stream_list):,} total instances from data/india_electricity_hourly.csv")

    val_len = 6000  # 250 days calibration warmup
    test_stream = stream_list[val_len:]
    print(f"Validation / Warmup: {val_len:,} instances | Test Stream: {len(test_stream):,} instances\n")

    # 1. Evaluate Core Models on Indian Test Stream
    print("[1/5] Running Core Models on Indian Grid Test Stream...")
    
    # Always-HT
    ht = HoeffdingTreeWrapper(grace_period=200)
    for x, y, _ in stream_list[:val_len]:
        ht.learn_one(x, y)
    res_ht = evaluate_stream(ht, test_stream, is_timed=True)
    print(f"  Always-HT:   Acc = {res_ht['accuracy_pct']}%, Mean Latency = {res_ht['mean_cpu_us']} us")

    # CGT
    cgt = CGTv3LearnedGateController(
        tau_gate=0.25, max_rho=0.12, h_cooldown=50, buffer_b=200, adwin_delta=0.005,
        arf_model=ARF3Wrapper(grace_period=200, lambda_value=6, delta=0.001, seed=42),
        seed=42
    )
    for x, y, _ in stream_list[:val_len]:
        p = cgt.predict_one(x)
        cgt.learn_one(x, y, p)
    res_cgt = evaluate_stream(cgt, test_stream, is_timed=True)
    rho_cgt = res_cgt["duty_cycle"]
    print(f"  CGT:         Acc = {res_cgt['accuracy_pct']}%, Duty = {rho_cgt*100:.2f}%, Mean Latency = {res_cgt['mean_cpu_us']} us")

    # ARF-3
    arf3 = ARF3Wrapper(grace_period=200, lambda_value=6, delta=0.001, seed=42)
    for x, y, _ in stream_list[:val_len]:
        arf3.learn_one(x, y)
    res_arf3 = evaluate_stream(arf3, test_stream, is_timed=True)
    print(f"  ARF-3 (3T):  Acc = {res_arf3['accuracy_pct']}%, Duty = 100%, Mean Latency = {res_arf3['mean_cpu_us']} us")

    # Always-ARF (Full 10 Trees)
    arf10 = ARFFullWrapper(n_models=10, seed=42)
    for x, y, _ in stream_list[:val_len]:
        arf10.learn_one(x, y)
    res_arf10 = evaluate_stream(arf10, test_stream, is_timed=True)
    print(f"  ARF-10 (10T): Acc = {res_arf10['accuracy_pct']}%, Duty = 100%, Mean Latency = {res_arf10['mean_cpu_us']} us")

    # OMS-MAB
    oms = OMSMABController(gamma=0.98, c=0.5, seed=42)
    for x, y, _ in stream_list[:val_len]:
        p = oms.predict_one(x)
        oms.learn_one(x, y, p)
    res_oms = evaluate_stream(oms, test_stream, is_timed=True)
    print(f"  OMS-MAB:     Acc = {res_oms['accuracy_pct']}%, Duty = {res_oms['duty_cycle']*100:.2f}%, Mean Latency = {res_oms['mean_cpu_us']} us")

    # Random-Gate at rho = rho_cgt (20 seeds)
    print("\n[2/5] Evaluating Random-Gate across 20 seeds at matching duty cycle...")
    n_blocks = 120  # 120 weeks x 168 hours = 20,160 hours
    block_size = 168
    eval_len = n_blocks * block_size  # 20,160 instances

    rg_accs = []
    rg_block_matrix = np.zeros((20, n_blocks))
    for s_idx, s in enumerate(range(1, 21)):
        rg = RandomGateBaseline(p=rho_cgt, seed=s)
        for x, y, _ in stream_list[:val_len]:
            p = rg.predict_one(x)
            rg.learn_one(x, y)
        res_s = evaluate_stream(rg, test_stream[:eval_len], is_timed=False)
        rg_accs.append(res_s["accuracy_pct"])
        corr = (res_s["y_preds"] == res_s["y_trues"]).astype(float)
        rg_block_matrix[s_idx, :] = corr.reshape(n_blocks, block_size).mean(axis=1) * 100.0

    mean_rg_acc = float(np.mean(rg_accs))
    std_rg_acc = float(np.std(rg_accs, ddof=1))
    mean_rg_blocks = np.mean(rg_block_matrix, axis=0)
    print(f"  Random-Gate (20 seeds) Acc: {mean_rg_acc:.3f}% +/- {std_rg_acc:.3f}%")

    # Save master Indian test summary
    df_india_summary = pd.DataFrame([
        {
            "Model": "Always-HT (Baseline)",
            "Ensemble Size": 1,
            "Duty Cycle (%)": 0.00,
            "Test Accuracy (%)": res_ht["accuracy_pct"],
            "Mean Latency (us)": res_ht["mean_cpu_us"],
            "Worst-case Latency (us)": res_ht["latency_max_us"],
            "Gating Nature": "Always Closed (0% Heavy)"
        },
        {
            "Model": f"Random-Gate (p={rho_cgt:.1%}, 20 seeds)",
            "Ensemble Size": "1 / 3",
            "Duty Cycle (%)": round(rho_cgt * 100.0, 2),
            "Test Accuracy (%)": round(mean_rg_acc, 3),
            "Mean Latency (us)": round(res_ht["mean_cpu_us"] + rho_cgt * res_arf3["mean_cpu_us"], 2),
            "Worst-case Latency (us)": res_arf3["latency_max_us"],
            "Gating Nature": "Pure Bernoulli Random Escalation"
        },
        {
            "Model": "CGT (Learned Meta-Gate)",
            "Ensemble Size": "1 / 3",
            "Duty Cycle (%)": round(res_cgt["duty_cycle"] * 100.0, 2),
            "Test Accuracy (%)": res_cgt["accuracy_pct"],
            "Mean Latency (us)": res_cgt["mean_cpu_us"],
            "Worst-case Latency (us)": res_cgt["latency_max_us"],
            "Gating Nature": "Selective Margin + ADWIN Cooldown"
        },
        {
            "Model": "ARF-3 (3 Trees)",
            "Ensemble Size": 3,
            "Duty Cycle (%)": 100.00,
            "Test Accuracy (%)": res_arf3["accuracy_pct"],
            "Mean Latency (us)": res_arf3["mean_cpu_us"],
            "Worst-case Latency (us)": res_arf3["latency_max_us"],
            "Gating Nature": "Always Open (100% 3-Tree Ensemble)"
        },
        {
            "Model": "OMS-MAB (Wilson et al., 2024)",
            "Ensemble Size": "1 / 10",
            "Duty Cycle (%)": round(res_oms["duty_cycle"] * 100.0, 2),
            "Test Accuracy (%)": res_oms["accuracy_pct"],
            "Mean Latency (us)": res_oms["mean_cpu_us"],
            "Worst-case Latency (us)": res_oms["latency_max_us"],
            "Gating Nature": "Unconstrained Multi-Armed Bandit"
        },
        {
            "Model": "Always-ARF (Full 10 Trees)",
            "Ensemble Size": 10,
            "Duty Cycle (%)": 100.00,
            "Test Accuracy (%)": res_arf10["accuracy_pct"],
            "Mean Latency (us)": res_arf10["mean_cpu_us"],
            "Worst-case Latency (us)": res_arf10["latency_max_us"],
            "Gating Nature": "Always Open (100% Heavy Ensemble)"
        }
    ])
    os.makedirs("results/tables", exist_ok=True)
    df_india_summary.to_csv("results/tables/india_fair_test_results.csv", index=False)
    print("\nIndian Grid Test Summary Table:")
    print(df_india_summary.to_string(index=False))

    # [3/5] 120-Week Block Hypothesis Testing on Indian Stream (Holm-Corrected)
    print("\n[3/5] Running 120-Week Block Statistical Significance Tests on Indian Stream...")
    y_test_eval = res_cgt["y_trues"][:eval_len]
    cgt_eval_corr = (res_cgt["y_preds"][:eval_len] == y_test_eval).astype(float)
    ht_eval_corr = (res_ht["y_preds"][:eval_len] == y_test_eval).astype(float)
    arf3_eval_corr = (res_arf3["y_preds"][:eval_len] == y_test_eval).astype(float)

    cgt_blocks = cgt_eval_corr.reshape(n_blocks, block_size).mean(axis=1) * 100.0
    ht_blocks = ht_eval_corr.reshape(n_blocks, block_size).mean(axis=1) * 100.0
    arf3_blocks = arf3_eval_corr.reshape(n_blocks, block_size).mean(axis=1) * 100.0
    rg_blocks = mean_rg_blocks

    comparisons = [
        ("CGT vs Always-HT", cgt_blocks - ht_blocks),
        ("CGT vs Random-Gate (20 seeds)", cgt_blocks - rg_blocks),
        ("ARF-3 vs CGT", arf3_blocks - cgt_blocks)
    ]

    stat_records = []
    raw_pvals = []
    for comp_name, diff in comparisons:
        m_diff = float(np.mean(diff))
        s_diff = float(np.std(diff, ddof=1))
        d_val = m_diff / s_diff if s_diff > 0 else 0.0
        w_stat, w_p = stats.wilcoxon(diff[diff != 0])
        t_stat, t_p = stats.ttest_1samp(diff, 0.0)
        raw_pvals.append(w_p)
        stat_records.append({
            "Comparison": comp_name,
            "Mean_Weekly_Delta_pct": round(m_diff, 3),
            "Paired_Cohen_d": round(d_val, 3),
            "Wilcoxon_W": round(float(w_stat), 2),
            "Raw_Wilcoxon_p": w_p,
            "Paired_t_stat": round(float(t_stat), 2),
            "Paired_t_p": t_p
        })

    # Holm-Bonferroni correction
    m = len(raw_pvals)
    sorted_idx = np.argsort(raw_pvals)
    holm_p = np.zeros(m)
    r_max = 0.0
    for rank, idx in enumerate(sorted_idx):
        adj = min(raw_pvals[idx] * (m - rank), 1.0)
        adj = max(adj, r_max)
        r_max = adj
        holm_p[idx] = adj

    for i, rec in enumerate(stat_records):
        rec["Holm_Adjusted_Wilcoxon_p"] = f"{holm_p[i]:.4e}"
        rec["Raw_Wilcoxon_p_str"] = f"{rec['Raw_Wilcoxon_p']:.4e}"
        rec["Significant_at_0.05"] = bool(holm_p[i] < 0.05)

    df_india_stats = pd.DataFrame(stat_records)
    df_india_stats.to_csv("results/tables/india_weekly_block_statistical_tests.csv", index=False)
    print("\nIndian Stream Weekly Block Tests (Holm-Corrected):")
    print(df_india_stats[["Comparison", "Mean_Weekly_Delta_pct", "Paired_Cohen_d", "Wilcoxon_W", "Raw_Wilcoxon_p_str", "Holm_Adjusted_Wilcoxon_p", "Significant_at_0.05"]].to_string(index=False))

    # [4/5] Multi-Rate Hardware Profiles at 50 Hz, 200 Hz, 1 kHz on Indian Stream
    print("\n[4/5] Computing Modeled Multi-Rate Hardware Profiles on Indian Stream...")
    models_hw = [
        {"name": "Always-HT", "lat_mean": res_ht["mean_cpu_us"], "lat_max": res_ht["latency_max_us"]},
        {"name": "Random-Gate (rho_CGT)", "lat_mean": round(res_ht["mean_cpu_us"] + rho_cgt * res_arf3["mean_cpu_us"], 2), "lat_max": res_arf3["latency_max_us"]},
        {"name": "CGT (Learned Gate)", "lat_mean": res_cgt["mean_cpu_us"], "lat_max": res_cgt["latency_max_us"]},
        {"name": "ARF-3 (3 Trees)", "lat_mean": res_arf3["mean_cpu_us"], "lat_max": res_arf3["latency_max_us"]},
        {"name": "OMS-MAB (Wilson et al.)", "lat_mean": res_oms["mean_cpu_us"], "lat_max": res_oms["latency_max_us"]},
        {"name": "Always-ARF (Full 10)", "lat_mean": res_arf10["mean_cpu_us"], "lat_max": res_arf10["latency_max_us"]}
    ]
    rates = [
        {"name": "50 Hz (Standard PMU)", "freq": 50.0, "t_us": 20000.0},
        {"name": "200 Hz (High-Speed PMU / DFR)", "freq": 200.0, "t_us": 5000.0},
        {"name": "1 kHz (Substation RTU Deadline)", "freq": 1000.0, "t_us": 1000.0}
    ]

    hw_rows = []
    for r in rates:
        t_per = r["t_us"]
        for m_info in models_hw:
            tm = m_info["lat_mean"]
            tw = m_info["lat_max"]
            duty_cpu = (tm / t_per) * 100.0
            
            if tm > t_per:
                sched = "OVERRUN (Mean Overload)"
            elif tw > t_per:
                sched = "OVERRUN (Replay Spike)"
            else:
                sched = "GUARANTEED"

            d_frac = min(duty_cpu / 100.0, 1.0)
            p_stm = d_frac * 280.0 + (1.0 - d_frac) * 8.2
            bat_stm = 100.0 / (p_stm / 1000.0)
            p_cm = d_frac * 3000.0 + (1.0 - d_frac) * 850.0
            bat_cm = 100.0 / (p_cm / 1000.0)

            hw_rows.append({
                "Sampling_Rate": r["name"],
                "Model_Name": m_info["name"],
                "Mean_Latency_us": tm,
                "Worst_Case_Latency_us": tw,
                "Modeled_CPU_Duty_pct": round(duty_cpu, 3),
                "RealTime_Schedulability (modeled)": sched,
                "STM32H743_Power_mW (modeled)": round(p_stm, 2),
                "STM32H743_100Wh_Autonomy_hrs (modeled)": round(bat_stm, 1),
                "CM4_100Wh_Autonomy_hrs (modeled)": round(bat_cm, 1)
            })

    df_hw = pd.DataFrame(hw_rows)
    df_hw.to_csv("results/tables/india_modeled_hardware_multirate.csv", index=False)
    print(df_hw[["Sampling_Rate", "Model_Name", "Mean_Latency_us", "Modeled_CPU_Duty_pct", "RealTime_Schedulability (modeled)", "STM32H743_100Wh_Autonomy_hrs (modeled)"]].to_string(index=False))

    # [5/5] Concept Drift Disaggregation on Indian Stream
    print("\n[5/5] Disaggregating Accuracy Inside vs. Outside Indian Grid Drift Regimes...")
    # Indian grid drift windows:
    # 1. Southwest Monsoon (July - August: cloud cover dampens solar, intense wind surges)
    # 2. Summer Heatwave Peaks (May - June: extreme AC peak load)
    # 3. Frequency Excursions (< 49.85 Hz or > 50.15 Hz, outside CEA statutory band)
    is_drift = np.zeros(len(test_stream), dtype=bool)
    for idx, (x_t, y_t, ts) in enumerate(test_stream):
        m = ts.month
        is_heatwave = (m in [5, 6])
        is_monsoon = (m in [7, 8])
        f_val = x_t.get('freq_lag_1', 50.0)
        is_freq_excursion = (f_val < 49.85 or f_val > 50.15)
        if is_heatwave or is_monsoon or is_freq_excursion:
            is_drift[idx] = True

    n_d = int(np.sum(is_drift))
    n_s = len(test_stream) - n_d
    print(f"  Indian Drift/Shock Regime: {n_d} hours ({n_d / len(test_stream) * 100:.2f}%)")
    print(f"  Indian Stationary Regime:   {n_s} hours ({n_s / len(test_stream) * 100:.2f}%)")

    drift_rows = []
    models_to_drift = [
        ("Always-HT", res_ht),
        ("CGT (Learned Gate)", res_cgt),
        ("ARF-3 (3 Trees)", res_arf3),
        ("OMS-MAB", res_oms),
        ("Always-ARF (Full 10)", res_arf10)
    ]
    for m_name, res in models_to_drift:
        corr_arr = (res["y_preds"] == res["y_trues"]).astype(float)
        acc_in = float(np.mean(corr_arr[is_drift])) * 100.0
        acc_out = float(np.mean(corr_arr[~is_drift])) * 100.0
        drift_rows.append({
            "Model_Name": m_name,
            "Overall_Acc_pct": res["accuracy_pct"],
            "Inside_Drift_Acc_pct": round(acc_in, 2),
            "Outside_Drift_Acc_pct": round(acc_out, 2),
            "Delta_Inside_Minus_Outside_pct": round(acc_in - acc_out, 2)
        })
    df_drift = pd.DataFrame(drift_rows)
    df_drift.to_csv("results/tables/india_drift_window_disaggregation.csv", index=False)
    print(df_drift.to_string(index=False))

    # Gap closed calculations
    gap_ht_to_arf10 = res_arf10["accuracy_pct"] - res_ht["accuracy_pct"]
    cgt_gain = res_cgt["accuracy_pct"] - res_ht["accuracy_pct"]
    pct_gap_arf10 = (cgt_gain / gap_ht_to_arf10) * 100.0

    gap_ht_to_arf3 = res_arf3["accuracy_pct"] - res_ht["accuracy_pct"]
    pct_gap_arf3 = (cgt_gain / gap_ht_to_arf3) * 100.0

    print("\n" + "=" * 80)
    print("GAP CLOSED SUMMARY ON INDIAN GRID (POSOCO / NLDC):")
    print(f"  Always-HT Acc:       {res_ht['accuracy_pct']}%")
    print(f"  CGT Acc:             {res_cgt['accuracy_pct']}% (Gain: +{cgt_gain:.2f} pp)")
    print(f"  ARF-3 Acc:           {res_arf3['accuracy_pct']}%")
    print(f"  Always-ARF (10T) Acc:{res_arf10['accuracy_pct']}%")
    print(f"  % of Gap Closed to Always-ARF (10T): {pct_gap_arf10:.2f}% ({cgt_gain:.2f} pp / {gap_ht_to_arf10:.2f} pp)")
    print(f"  % of Gap Closed to ARF-3 (3T):       {pct_gap_arf3:.2f}% ({cgt_gain:.2f} pp / {gap_ht_to_arf3:.2f} pp)")
    print("=" * 80)

if __name__ == "__main__":
    main()

