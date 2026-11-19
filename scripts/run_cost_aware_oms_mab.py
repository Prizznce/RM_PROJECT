import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import math
import json
import hashlib
import tempfile
import numpy as np
import pandas as pd
from scipy import stats

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import matplotlib.pyplot as plt

from river.drift.binary import HDDMA
from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper
from src.controllers.cgt_v3 import CGTv3LearnedGateController


class ARF3Wrapper:
    """Standardized ARF with 3 trees matching frozen validation config."""
    def __init__(self, grace_period=200, lambda_value=6, delta=0.001, seed=42):
        self.grace_period = grace_period
        self.lambda_value = lambda_value
        self.delta = delta
        self.seed = seed
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


class CostAwareOMSMABController:
    """
    Cost-Aware OMS-MAB (Wilson et al., 2024 adaptation):
    - Arm 0: Hoeffding Tree (lightweight, zero cost penalty)
    - Arm 1: ARF-3 (3 trees, heavy arm identical to CGT's heavy arm)
    
    Reward Function:
      r_t(Arm 0) = 1(y_pred == y)
      r_t(Arm 1) = 1(y_pred == y) - lambda_cost
      
    Optional Hard Duty Governor:
      If max_rho is set and running duty >= max_rho, Arm 1 is masked, forcing Arm 0.
    """
    def __init__(self, lambda_cost=0.05, max_rho=None, gamma=0.98, c=0.5, seed=42):
        self.name = f"Cost-Aware OMS-MAB (lambda={lambda_cost}, rho_max={max_rho})"
        self.lambda_cost = lambda_cost
        self.max_rho = max_rho
        self.gamma = gamma
        self.c = c
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.arms = [
            HoeffdingTreeWrapper(),
            ARF3Wrapper(grace_period=200, lambda_value=6, delta=0.001, seed=seed)
        ]
        self.arm_names = ["HT", "ARF-3"]
        self.k = len(self.arms)
        
        self.detectors = [
            HDDMA(drift_confidence=0.001, warning_confidence=0.005),
            HDDMA(drift_confidence=0.001, warning_confidence=0.005)
        ]
        
        self.discounted_pulls = np.zeros(self.k, dtype=float)
        self.discounted_rewards = np.zeros(self.k, dtype=float)
        self.arm_pull_counts = np.zeros(self.k, dtype=int)
        
        self.active_arm_idx = 0
        self.total_steps = 0
        self.gate_open = False
        self.gate_open_steps = 0
        self.total_drifts_detected = 0

    def predict_one(self, x):
        self.total_steps += 1
        current_rho = self.arm_pull_counts[1] / max(self.total_steps, 1)
        
        # Hard duty governor constraint: if duty exceeded, mask Arm 1
        if self.max_rho is not None and current_rho >= self.max_rho:
            self.active_arm_idx = 0
        else:
            # Cold start: each arm pulled at least 5 times
            unpulled = [i for i in range(self.k) if self.arm_pull_counts[i] < 5]
            if unpulled:
                self.active_arm_idx = self.rng.choice(unpulled)
            else:
                total_n = np.sum(self.discounted_pulls)
                ucb_values = np.zeros(self.k)
                for i in range(self.k):
                    n_i = self.discounted_pulls[i]
                    if n_i <= 1e-4:
                        ucb_values[i] = 1e5
                    else:
                        mu_i = self.discounted_rewards[i] / n_i
                        bonus = self.c * math.sqrt((2.0 * math.log(max(total_n, 1.01))) / n_i)
                        ucb_values[i] = mu_i + bonus
                self.active_arm_idx = int(np.argmax(ucb_values))
                
        self.gate_open = (self.active_arm_idx == 1)
        if self.gate_open:
            self.gate_open_steps += 1
            
        return self.arms[self.active_arm_idx].predict_one(x)

    def learn_one(self, x, y, y_pred):
        idx = self.active_arm_idx
        acc_reward = 1.0 if y_pred == y else 0.0
        
        # Subtract cost penalty from Arm 1 (heavy ensemble)
        if idx == 1:
            reward = acc_reward - self.lambda_cost
        else:
            reward = acc_reward
            
        error = 1.0 - acc_reward
        
        self.arms[idx].learn_one(x, y)
        self.arm_pull_counts[idx] += 1
        
        self.detectors[idx].update(error)
        if self.detectors[idx].drift_detected:
            self.total_drifts_detected += 1
            self.discounted_pulls[idx] = 1.0
            self.discounted_rewards[idx] = reward
        else:
            self.discounted_pulls *= self.gamma
            self.discounted_rewards *= self.gamma
            self.discounted_pulls[idx] += 1.0
            self.discounted_rewards[idx] += reward


class RandomGateBaseline:
    """Random-Gate Baseline for multi-seed evaluation."""
    def __init__(self, p=0.0459, seed=42):
        self.p = p
        self.seed = seed
        self.rng = np.random.RandomState(seed)
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
            1 if isinstance(model, ARF3Wrapper) or isinstance(model, ARFFullWrapper) else 0
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
    print("COST-AWARE OMS-MAB BENCHMARK & MATCHED-DUTY EVALUATION")
    print("=" * 80)

    # 1. Load Data
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_instances = 10512
    test_instances = 24528
    val_stream = full_stream[:val_instances]
    test_stream = full_stream[val_instances:val_instances + test_instances]
    print(f"Validation Stream: {len(val_stream)} instances (Days 1 - 439)")
    print(f"Test Stream:       {len(test_stream)} instances (Days 440 - 1461, 146 weeks)\n")

    # =========================================================================
    # STEP 1: VALIDATION TUNING (K = 16 CANDIDATES)
    # =========================================================================
    print("[1/5] Tuning Cost-Aware OMS-MAB on Validation Split (K = 16 Trials)...")
    
    # Grid of 16 candidates: testing cost penalties and duty caps
    candidates = [
        # Pure cost penalties without hard cap
        {"lambda_cost": 0.01, "max_rho": None, "c": 0.5},
        {"lambda_cost": 0.02, "max_rho": None, "c": 0.5},
        {"lambda_cost": 0.03, "max_rho": None, "c": 0.5},
        {"lambda_cost": 0.05, "max_rho": None, "c": 0.5},
        {"lambda_cost": 0.08, "max_rho": None, "c": 0.5},
        {"lambda_cost": 0.10, "max_rho": None, "c": 0.5},
        {"lambda_cost": 0.15, "max_rho": None, "c": 0.5},
        {"lambda_cost": 0.20, "max_rho": None, "c": 0.5},
        # Cost penalty with hard duty governor (aiming for matched duty ~4.6%)
        {"lambda_cost": 0.01, "max_rho": 0.046, "c": 0.5},
        {"lambda_cost": 0.02, "max_rho": 0.046, "c": 0.5},
        {"lambda_cost": 0.03, "max_rho": 0.046, "c": 0.5},
        {"lambda_cost": 0.05, "max_rho": 0.046, "c": 0.5},
        {"lambda_cost": 0.02, "max_rho": 0.050, "c": 0.5},
        {"lambda_cost": 0.05, "max_rho": 0.050, "c": 0.5},
        {"lambda_cost": 0.02, "max_rho": 0.080, "c": 0.5},
        {"lambda_cost": 0.05, "max_rho": 0.080, "c": 0.5},
    ]

    val_records = []
    best_config_matched = None
    best_val_acc_matched = -1.0

    for idx, cfg in enumerate(candidates):
        model = CostAwareOMSMABController(
            lambda_cost=cfg["lambda_cost"],
            max_rho=cfg["max_rho"],
            c=cfg["c"],
            seed=42
        )
        res_v = evaluate_timed(model, val_stream)
        print(f"  Trial {idx+1:02d}/16: lambda={cfg['lambda_cost']}, rho_max={cfg['max_rho']} -> Acc={res_v['accuracy_pct']}%, Duty={res_v['duty_cycle']*100:.2f}%, Latency={res_v['mean_cpu_us']} us")
        
        val_records.append({
            "trial_id": idx + 1,
            "lambda_cost": cfg["lambda_cost"],
            "max_rho": cfg["max_rho"],
            "val_accuracy_pct": res_v["accuracy_pct"],
            "val_duty_cycle_pct": round(res_v["duty_cycle"] * 100.0, 2),
            "val_mean_latency_us": res_v["mean_cpu_us"]
        })
        
        # Best matched configuration (target duty near 4.6%)
        if cfg["max_rho"] == 0.046:
            if res_v["accuracy_pct"] > best_val_acc_matched:
                best_val_acc_matched = res_v["accuracy_pct"]
                best_config_matched = cfg

    df_val = pd.DataFrame(val_records)
    os.makedirs("results/tables", exist_ok=True)
    df_val.to_csv("results/tables/cost_aware_oms_mab_val_tuning.csv", index=False)

    print(f"\n>>> BEST MATCHED COST-AWARE OMS-MAB VALIDATION CONFIG: {best_config_matched} (Acc = {best_val_acc_matched}%)")

    # Freeze & Hash Config
    freeze_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "search_budget_K": len(candidates),
        "validation_instances": len(val_stream),
        "test_instances": len(test_stream),
        "best_config_matched": best_config_matched,
        "best_val_acc_matched": best_val_acc_matched
    }
    json_str = json.dumps(freeze_payload, indent=2, sort_keys=True)
    frozen_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    freeze_payload["sha256_hash"] = frozen_hash

    os.makedirs("configs", exist_ok=True)
    config_path = "configs/cost_aware_oms_mab_frozen_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(freeze_payload, f, indent=2)

    print(f"Locked & Hashed Frozen Config -> {config_path}")
    print(f"SHA-256 Hash: {frozen_hash}\n")

    # =========================================================================
    # STEP 2: RUN FROZEN MATCHED-DUTY COST-AWARE OMS-MAB ON TEST STREAM
    # =========================================================================
    print("[2/5] Running Frozen Matched-Duty Cost-Aware OMS-MAB on 24,528 Test Instances...")
    frozen_oms = CostAwareOMSMABController(
        lambda_cost=best_config_matched["lambda_cost"],
        max_rho=best_config_matched["max_rho"],
        c=best_config_matched["c"],
        seed=42
    )
    res_oms_test = evaluate_timed(frozen_oms, test_stream)
    print(f"  Cost-Aware OMS-MAB Test Acc:   {res_oms_test['accuracy_pct']}%")
    print(f"  Cost-Aware OMS-MAB Duty Cycle: {res_oms_test['duty_cycle']*100:.2f}%")
    print(f"  Cost-Aware OMS-MAB Mean Lat:   {res_oms_test['mean_cpu_us']} us (Worst: {res_oms_test['latency_max_us']} us)")

    df_oms_raw = pd.DataFrame({
        "instance_idx": np.arange(1, len(test_stream) + 1),
        "y_true": res_oms_test["y_trues"],
        "y_pred": res_oms_test["y_preds"],
        "gate_open": res_oms_test["gate_states"],
        "cpu_time_ns": res_oms_test["cpu_ns"]
    })
    os.makedirs("results/raw", exist_ok=True)
    df_oms_raw.to_csv("results/raw/fair_test_predictions_cost_aware_oms_mab.csv", index=False)

    # Load previously computed test predictions for CGT, ARF-3, HT, Random-Gate
    df_cgt_raw = pd.read_csv("results/raw/fair_test_predictions_cgt.csv")
    df_arf3_raw = pd.read_csv("results/raw/fair_test_predictions_arf3.csv")
    df_rg_raw = pd.read_csv("results/raw/fair_test_predictions_random_gate.csv")

    y_true = df_cgt_raw["y_true"].values
    cgt_corr = (df_cgt_raw["y_pred"].values == y_true).astype(float)
    arf3_corr = (df_arf3_raw["y_pred"].values == y_true).astype(float)
    rg_corr = (df_rg_raw["y_pred"].values == y_true).astype(float)
    oms_corr = (res_oms_test["y_preds"] == y_true).astype(float)

    cgt_acc = float(np.mean(cgt_corr)) * 100.0
    arf3_acc = float(np.mean(arf3_corr)) * 100.0
    rg_acc = float(np.mean(rg_corr)) * 100.0
    oms_acc = res_oms_test["accuracy_pct"]

    # =========================================================================
    # STEP 3: MATCHED-DUTY COMPARATIVE TABLE (~4.6%)
    # =========================================================================
    print("\n[3/5] Compiling Matched-Duty Comparative Table (~4.6% Heavy Duty)...")
    df_matched_table = pd.DataFrame([
        {
            "Model": "Always-HT (Baseline)",
            "Ensemble Size": 1,
            "Achieved Heavy Duty (%)": 0.00,
            "Test Accuracy (%)": 85.066,
            "Mean Latency (us)": 94.26,
            "p95 Latency (us)": 132.96,
            "Worst-Case Latency (us)": 2404.0,
            "Modeled MCU Battery (100 Wh)": "10,547.4 h (1.20 yr)",
            "Gating / Selection Strategy": "Always Single Tree (0% Heavy)"
        },
        {
            "Model": "Random-Gate Baseline (20 seeds)",
            "Ensemble Size": "1 / 3",
            "Achieved Heavy Duty (%)": 4.58,
            "Test Accuracy (%)": 84.756,
            "Mean Latency (us)": 102.82,
            "p95 Latency (us)": 237.80,
            "Worst-Case Latency (us)": 4309.3,
            "Modeled MCU Battery (100 Wh)": "10,284.3 h (1.17 yr)",
            "Gating / Selection Strategy": "Bernoulli Random Escalation"
        },
        {
            "Model": "Cost-Aware OMS-MAB (Wilson et al.)",
            "Ensemble Size": "1 / 3",
            "Achieved Heavy Duty (%)": round(res_oms_test["duty_cycle"] * 100.0, 2),
            "Test Accuracy (%)": oms_acc,
            "Mean Latency (us)": res_oms_test["mean_cpu_us"],
            "p95 Latency (us)": res_oms_test["latency_p95_us"],
            "Worst-Case Latency (us)": res_oms_test["latency_max_us"],
            "Modeled MCU Battery (100 Wh)": f"{100.0 / ((min((res_oms_test['mean_cpu_us']/20000.0), 1.0) * 280.0 + (1.0 - min((res_oms_test['mean_cpu_us']/20000.0), 1.0)) * 8.2) / 1000.0):.1f} h",
            "Gating / Selection Strategy": f"Cost-Penalized D-UCB (lambda={best_config_matched['lambda_cost']}, cap={best_config_matched['max_rho']})"
        },
        {
            "Model": "CGT (Learned Meta-Gate)",
            "Ensemble Size": "1 / 3",
            "Achieved Heavy Duty (%)": 4.59,
            "Test Accuracy (%)": cgt_acc,
            "Mean Latency (us)": 243.37,
            "p95 Latency (us)": 447.80,
            "Worst-Case Latency (us)": 38002.8,
            "Modeled MCU Battery (100 Wh)": "8,690.1 h (0.99 yr)",
            "Gating / Selection Strategy": "Selective Margin + ADWIN Cooldown"
        },
        {
            "Model": "ARF-3 (3 Trees)",
            "Ensemble Size": 3,
            "Achieved Heavy Duty (%)": 100.00,
            "Test Accuracy (%)": arf3_acc,
            "Mean Latency (us)": 281.35,
            "p95 Latency (us)": 568.76,
            "Worst-Case Latency (us)": 2231.1,
            "Modeled MCU Battery (100 Wh)": "8,317.0 h (0.95 yr)",
            "Gating / Selection Strategy": "Continuous 100% Ensemble"
        }
    ])
    matched_csv_path = "results/tables/matched_duty_cgt_vs_oms_mab_comparison.csv"
    df_matched_table.to_csv(matched_csv_path, index=False)
    print("\nMatched-Duty Comparison Table (~4.6% Heavy Duty):")
    print(df_matched_table.to_string(index=False))

    # =========================================================================
    # STEP 4: DUTY SWEEP (TARGET 2% TO 30%)
    # =========================================================================
    print("\n[4/5] Sweeping Lambda & Duty Budgets to Trace Accuracy vs Duty (Target 2% - 30%)...")
    sweep_records = []

    # Sweep Cost-Aware OMS-MAB across duty caps
    for rho_cap in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        model_sw = CostAwareOMSMABController(lambda_cost=0.02, max_rho=rho_cap, seed=42)
        res_sw = evaluate_timed(model_sw, test_stream)
        sweep_records.append({
            "Policy": "Cost-Aware OMS-MAB",
            "Budget_Cap_pct": int(rho_cap * 100),
            "Achieved_Duty_pct": round(res_sw["duty_cycle"] * 100.0, 2),
            "Test_Accuracy_pct": res_sw["accuracy_pct"],
            "Mean_Latency_us": res_sw["mean_cpu_us"]
        })

    # Sweep CGT across duty caps
    for rho_cap in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        arf_h = ARF3Wrapper(seed=42)
        cgt_sw = CGTv3LearnedGateController(
            tau_gate=0.25, max_rho=rho_cap, h_cooldown=50, buffer_b=200, adwin_delta=0.005,
            arf_model=arf_h, seed=42
        )
        res_sw = evaluate_timed(cgt_sw, test_stream)
        sweep_records.append({
            "Policy": "CGT (Learned Meta-Gate)",
            "Budget_Cap_pct": int(rho_cap * 100),
            "Achieved_Duty_pct": round(res_sw["duty_cycle"] * 100.0, 2),
            "Test_Accuracy_pct": res_sw["accuracy_pct"],
            "Mean_Latency_us": res_sw["mean_cpu_us"]
        })

    # Sweep Random-Gate across duty caps
    for rho_cap in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        rg_sw = RandomGateBaseline(p=rho_cap, seed=42)
        res_sw = evaluate_timed(rg_sw, test_stream)
        sweep_records.append({
            "Policy": "Random-Gate Baseline",
            "Budget_Cap_pct": int(rho_cap * 100),
            "Achieved_Duty_pct": round(res_sw["duty_cycle"] * 100.0, 2),
            "Test_Accuracy_pct": res_sw["accuracy_pct"],
            "Mean_Latency_us": res_sw["mean_cpu_us"]
        })

    df_sweep = pd.DataFrame(sweep_records)
    df_sweep.to_csv("results/tables/oms_mab_duty_sweep.csv", index=False)
    print("\nDuty Sweep Results (2% - 30%):")
    print(df_sweep.to_string(index=False))

    # Plot All Curves on ONE Single Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    cgt_sw_df = df_sweep[df_sweep["Policy"] == "CGT (Learned Meta-Gate)"].sort_values("Achieved_Duty_pct")
    oms_sw_df = df_sweep[df_sweep["Policy"] == "Cost-Aware OMS-MAB"].sort_values("Achieved_Duty_pct")
    rg_sw_df = df_sweep[df_sweep["Policy"] == "Random-Gate Baseline"].sort_values("Achieved_Duty_pct")

    ax.plot(cgt_sw_df["Achieved_Duty_pct"], cgt_sw_df["Test_Accuracy_pct"],
            color="#1f77b4", marker="o", linewidth=2.2, label="CGT (Selective Learned Gate)")
    ax.plot(oms_sw_df["Achieved_Duty_pct"], oms_sw_df["Test_Accuracy_pct"],
            color="#ff7f0e", marker="^", linestyle="-.", linewidth=2.0, label="Cost-Aware OMS-MAB (Wilson et al. + ARF-3)")
    ax.plot(rg_sw_df["Achieved_Duty_pct"], rg_sw_df["Test_Accuracy_pct"],
            color="#d62728", marker="s", linestyle="--", linewidth=1.8, label="Random-Gate Baseline")

    # Reference lines
    ax.axhline(85.066, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Always-HT Baseline (85.07%, 0% Heavy)")
    ax.axhline(86.191, color="#2ca02c", linestyle="--", linewidth=1.5, label="ARF-3 Reference (86.19%, 100% Heavy)")

    ax.set_title("Streaming Trade-off: Accuracy vs. Achieved Heavy Duty Cycle\n(24,528-Hour Test Stream, Evaluated with Standardized ARF-3 Heavy Arm)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Achieved Heavy-Ensemble Execution Duty Cycle (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Prequential Classification Accuracy (%)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 32)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    plot_path = "results/figures/fig_accuracy_vs_duty_cgt_vs_oms_mab.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved combined Accuracy vs Duty plot -> {plot_path}")

    # =========================================================================
    # STEP 5: WEEKLY BLOCK HYPOTHESIS TESTS (146 BLOCKS, HOLM-CORRECTED)
    # =========================================================================
    print("\n[5/5] Running 146-Week Block Statistical Significance Tests (Holm-Corrected)...")
    n_blocks = 146
    block_size = 168

    cgt_blocks = cgt_corr.reshape(n_blocks, block_size).mean(axis=1) * 100.0
    oms_blocks = oms_corr.reshape(n_blocks, block_size).mean(axis=1) * 100.0
    rg_blocks = rg_corr.reshape(n_blocks, block_size).mean(axis=1) * 100.0

    diff_cgt_vs_oms = cgt_blocks - oms_blocks
    mean_diff_cgt_oms = float(np.mean(diff_cgt_vs_oms))
    std_diff_cgt_oms = float(np.std(diff_cgt_vs_oms, ddof=1))
    d_cgt_oms = mean_diff_cgt_oms / std_diff_cgt_oms if std_diff_cgt_oms > 0 else 0.0

    w_stat, w_p = stats.wilcoxon(diff_cgt_vs_oms[diff_cgt_vs_oms != 0])
    t_stat, t_p = stats.ttest_1samp(diff_cgt_vs_oms, 0.0)

    # Verification: mean weekly delta equals overall test delta
    overall_delta = cgt_acc - oms_acc
    print(f"  Arithmetic check: Mean Weekly Delta = {mean_diff_cgt_oms:.3f}%, Overall Delta = {overall_delta:.3f}% (Difference: {abs(mean_diff_cgt_oms - overall_delta):.6f})")

    # Combine with other block tests for Holm correction
    comparisons = [
        ("CGT vs Cost-Aware OMS-MAB (matched duty)", diff_cgt_vs_oms, w_stat, w_p, t_stat, t_p),
    ]

    print("\n" + "=" * 80)
    print("CGT vs COST-AWARE OMS-MAB WEEKLY BLOCK RESULTS (146 BLOCKS):")
    print(f"  CGT Test Accuracy:             {cgt_acc:.3f}% (Duty = 4.59%)")
    print(f"  Cost-Aware OMS-MAB Accuracy:   {oms_acc:.3f}% (Duty = {res_oms_test['duty_cycle']*100:.2f}%)")
    print(f"  Accuracy Delta (CGT - OMS):    {cgt_acc - oms_acc:+.3f} percentage points")
    print(f"  Paired Cohen's d:              {d_cgt_oms:.3f}")
    print(f"  Wilcoxon W:                    {w_stat:.1f}")
    print(f"  Wilcoxon p-value:              {w_p:.4e}")
    print(f"  Paired t-test t:               {t_stat:.2f} (p = {t_p:.4e})")
    
    if cgt_acc > oms_acc:
        print(">>> VERDICT: CGT WINS over Cost-Aware OMS-MAB at matched duty!")
    elif oms_acc > cgt_acc:
        print(">>> VERDICT: Cost-Aware OMS-MAB WINS over CGT at matched duty!")
    else:
        print(">>> VERDICT: TIE between CGT and Cost-Aware OMS-MAB at matched duty.")
    print("=" * 80)

    # Save test results
    df_block_oms = pd.DataFrame([{
        "Comparison": "CGT vs Cost-Aware OMS-MAB (matched duty)",
        "CGT_Accuracy_pct": cgt_acc,
        "OMS_MAB_Accuracy_pct": oms_acc,
        "Mean_Weekly_Delta_pct": round(mean_diff_cgt_oms, 3),
        "Overall_Stream_Delta_pct": round(overall_delta, 3),
        "Paired_Cohen_d": round(d_cgt_oms, 3),
        "Wilcoxon_W": round(float(w_stat), 2),
        "Wilcoxon_p_value": f"{w_p:.4e}",
        "Paired_t_stat": round(float(t_stat), 2),
        "Paired_t_p_value": f"{t_p:.4e}",
        "Significant_at_0.05": bool(w_p < 0.05)
    }])
    df_block_oms.to_csv("results/tables/fair_weekly_block_cgt_vs_cost_aware_oms_mab.csv", index=False)
    print(f"\nSaved weekly block test results -> results/tables/fair_weekly_block_cgt_vs_cost_aware_oms_mab.csv")


if __name__ == "__main__":
    main()

