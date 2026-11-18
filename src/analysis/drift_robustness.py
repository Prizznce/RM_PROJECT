"""
Task E: Empirical Drift & Stress Robustness Evaluation on an In-Memory Test-Stream COPY.
Evaluates 3 controlled shock scenarios:
  1. Heatwave demand surge (14 days = 336 hours at t = 4000)
  2. Dunkelflaute renewable drop (7 days = 168 hours at t = 10000)
  3. Permanent structural load level shift (t = 16000 to end)

Metrics per policy:
  - Accuracy during shock window (%)
  - Detection delay (steps to first alarm / gate activation)
  - ARF active duty cycle during shock window
  - Recovery time (steps until sliding accuracy recovers to within 1 point of pre-shock)

Figure:
  results/figures/fig_day7_drift_robustness.png showing gate dynamics and ADWIN alarms.
"""

import copy
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper
from src.controllers.cgt import ConditionalGatedTrainingController
from src.controllers.cgt_v2 import CGTv2Controller
from src.controllers.cgt_v3 import CGTv3LearnedGateController

def build_shocked_test_stream_copy():
    with open("configs/frozen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_len = cfg["data"]["val_instances"]

    # Strict deep-copy of test stream: original dataset is NEVER modified
    val_stream = full_stream[:val_len]
    test_stream = [copy.deepcopy(item) for item in full_stream[val_len:]]

    rng = np.random.RandomState(42)

    # Shock 1: Heatwave Demand Surge (14 days = 336 hours, steps 4000 to 4336)
    # Demand jumps by +30%, massive upward ramp events
    for i in range(4000, min(4336, len(test_stream))):
        x, y, ts = test_stream[i]
        x_mod = dict(x)
        x_mod["demand_lag_1"] = x_mod.get("demand_lag_1", 30000.0) * 1.30
        x_mod["demand_lag_2"] = x_mod.get("demand_lag_2", 30000.0) * 1.28
        x_mod["demand_lag_24"] = x_mod.get("demand_lag_24", 30000.0) * 1.15
        x_mod["demand_diff_1_2"] = x_mod["demand_lag_1"] - x_mod["demand_lag_2"]
        x_mod["demand_diff_1_24"] = x_mod["demand_lag_1"] - x_mod["demand_lag_24"]
        y_mod = 1 if rng.rand() < 0.85 else 0
        test_stream[i] = (x_mod, y_mod, ts)

    # Shock 2: Dunkelflaute Renewable Collapse (7 days = 168 hours, steps 10000 to 10168)
    # Solar and wind collapse by 85%, peaking gas increases
    for i in range(10000, min(10168, len(test_stream))):
        x, y, ts = test_stream[i]
        x_mod = dict(x)
        x_mod["solar_lag_1"] = x_mod.get("solar_lag_1", 2000.0) * 0.15
        x_mod["wind_lag_1"] = x_mod.get("wind_lag_1", 5000.0) * 0.15
        x_mod["gas_lag_1"] = x_mod.get("gas_lag_1", 4000.0) * 1.65
        x_mod["renewable_ratio_lag_1"] = (x_mod["solar_lag_1"] + x_mod["wind_lag_1"]) / max(x_mod.get("total_gen_lag_1", 30000.0), 1.0)
        # Higher label volatility due to uncoordinated peaking dispatch
        y_mod = 1 if (y == 0 and rng.rand() < 0.60) else y
        test_stream[i] = (x_mod, y_mod, ts)

    # Shock 3: Permanent Structural Load Level Shift (steps 16000 to end)
    for i in range(16000, len(test_stream)):
        x, y, ts = test_stream[i]
        x_mod = dict(x)
        x_mod["demand_lag_1"] = x_mod.get("demand_lag_1", 30000.0) * 1.20
        x_mod["demand_lag_2"] = x_mod.get("demand_lag_2", 30000.0) * 1.18
        x_mod["demand_diff_1_2"] = (x_mod["demand_lag_1"] - x_mod["demand_lag_2"]) * -1.2  # inverted momentum
        y_mod = 1 if x_mod["demand_diff_1_2"] > 0 else 0
        test_stream[i] = (x_mod, y_mod, ts)

    return val_stream, test_stream

def evaluate_shock_policy(policy, val_stream, shocked_test_stream):
    # Warm-up on clean validation stream
    for x, y, ts in val_stream:
        y_pred = policy.predict_one(x)
        try:
            policy.learn_one(x, y, y_pred)
        except TypeError:
            policy.learn_one(x, y)

    n_test = len(shocked_test_stream)
    correct_arr = np.zeros(n_test, dtype=int)
    gate_arr = np.zeros(n_test, dtype=int)
    alarm_indices = []

    for idx, (x, y, ts) in enumerate(shocked_test_stream):
        # Check alarm state before learn
        prev_alarms = getattr(policy, "total_alarms", 0)
        y_pred = policy.predict_one(x)
        try:
            policy.learn_one(x, y, y_pred)
        except TypeError:
            policy.learn_one(x, y)

        if getattr(policy, "total_alarms", 0) > prev_alarms:
            alarm_indices.append(idx)

        correct_arr[idx] = 1 if y_pred == y else 0
        gate_arr[idx] = 1 if getattr(policy, "gate_open", False) else 0

    return correct_arr, gate_arr, alarm_indices

def run_drift_robustness():
    print("=================================================================", flush=True)
    print("TASK E: EMPIRICAL DRIFT & STRESS ROBUSTNESS EVALUATION (TEST COPY)", flush=True)
    print("=================================================================\n", flush=True)

    val_stream, shocked_test_stream = build_shocked_test_stream_copy()
    print(f"Synthesized 3 controlled shocks on in-memory test stream ({len(shocked_test_stream)} instances).", flush=True)

    with open("configs/frozen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = 42

    policies = {
        "Always-HT": HoeffdingTreeWrapper(),
        "Always-ARF": ARFFullWrapper(seed=seed),
        "CGT-v1": ConditionalGatedTrainingController(
            stability_k=cfg["cgt_frozen"]["stability_k"],
            buffer_b=cfg["cgt_frozen"]["buffer_b"],
            warm_start_strategy="S1",
            adwin_delta=cfg["cgt_frozen"]["adwin_delta"],
            adwin_clock=cfg["cgt_frozen"]["adwin_clock"],
            seed=seed
        ),
        "CGT-v2": CGTv2Controller(
            theta_margin=0.25, max_rho=0.15, h_cooldown=100, buffer_b=200, use_pruned_arf=True, seed=seed
        ),
        "CGT-v3": CGTv3LearnedGateController(
            tau_gate=0.20, max_rho=0.10, h_cooldown=50, buffer_b=200, use_pruned_arf=True, seed=seed
        )
    }

    shock_specs = [
        {"name": "Shock 1: Heatwave (14d)", "start": 4000, "end": 4336, "pre_start": 3800, "pre_end": 4000},
        {"name": "Shock 2: Dunkelflaute (7d)", "start": 10000, "end": 10168, "pre_start": 9800, "pre_end": 10000},
        {"name": "Shock 3: Permanent Level Shift", "start": 16000, "end": 17000, "pre_start": 15800, "pre_end": 16000}
    ]

    sim_results = {}
    for p_name, policy in policies.items():
        print(f"Simulating stress response for {p_name}...", flush=True)
        correct_arr, gate_arr, alarm_indices = evaluate_shock_policy(policy, val_stream, shocked_test_stream)
        sim_results[p_name] = {
            "correct": correct_arr,
            "gate": gate_arr,
            "alarms": alarm_indices
        }

    # Calculate metrics across shock windows
    metrics_list = []
    for shock in shock_specs:
        s_name = shock["name"]
        s_start = shock["start"]
        s_end = shock["end"]
        pre_s = shock["pre_start"]
        pre_e = shock["pre_end"]

        for p_name in policies.keys():
            res = sim_results[p_name]
            corr = res["correct"]
            gate = res["gate"]
            alarms = res["alarms"]

            # 1. Pre-shock accuracy
            pre_acc = corr[pre_s:pre_e].mean() * 100.0

            # 2. Accuracy during shock
            shock_acc = corr[s_start:s_end].mean() * 100.0

            # 3. Detection delay: first alarm or gate awakening >= s_start
            delay = None
            if "CGT" in p_name:
                future_alarms = [a for a in alarms if a >= s_start]
                first_alarm_step = (future_alarms[0] - s_start) if future_alarms else None

                future_gates = np.where(gate[s_start:] == 1)[0]
                first_gate_step = int(future_gates[0]) if len(future_gates) > 0 else None

                delay = first_gate_step if first_gate_step is not None else first_alarm_step
            else:
                delay = "n/a"

            # 4. ARF active duty cycle during shock
            if p_name == "Always-ARF":
                arf_rho_shock = 1.0
            elif p_name == "Always-HT":
                arf_rho_shock = 0.0
            else:
                arf_rho_shock = float(gate[s_start:s_end].mean())

            # 5. Recovery time: steps after s_start until sliding accuracy (W=100) >= pre_acc - 1.0
            rec_steps = "No Recovery"
            target_acc = (pre_acc - 1.0) / 100.0
            win_w = 100
            for step in range(s_start + win_w, min(s_start + 2000, len(corr))):
                slide_acc = corr[step - win_w:step].mean()
                if slide_acc >= target_acc:
                    rec_steps = step - s_start
                    break

            metrics_list.append({
                "shock_scenario": s_name,
                "policy": p_name,
                "pre_shock_acc_pct": round(pre_acc, 2),
                "shock_window_acc_pct": round(shock_acc, 2),
                "acc_drop_pct": round(pre_acc - shock_acc, 2),
                "detection_delay_steps": delay if delay is not None else "None",
                "arf_duty_cycle_during_shock": round(arf_rho_shock, 4) if isinstance(arf_rho_shock, float) else arf_rho_shock,
                "recovery_time_steps": rec_steps
            })

    metrics_df = pd.DataFrame(metrics_list)
    print("\n--- Stress & Concept Drift Robustness Results ---")
    print(metrics_df[["shock_scenario", "policy", "shock_window_acc_pct", "acc_drop_pct", "detection_delay_steps", "arf_duty_cycle_during_shock", "recovery_time_steps"]].to_string(index=False))

    os.makedirs("results/tables", exist_ok=True)
    metrics_df.to_csv("results/tables/day7_drift_robustness_metrics.csv", index=False)
    print("\nSaved drift metrics table to results/tables/day7_drift_robustness_metrics.csv", flush=True)

    # Generate Publication Figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    plot_windows = [
        ("Shock 1: Heatwave Demand Surge (+30%)", 3800, 4600, axes[0], 4000, 4336),
        ("Shock 2: Dunkelflaute Renewable Collapse (-85% Solar/Wind)", 9800, 10400, axes[1], 10000, 10168),
        ("Shock 3: Permanent Structural Load Shift", 15800, 17200, axes[2], 16000, 17200)
    ]

    colors = {
        "Always-HT": "#1f77b4",
        "Always-ARF": "#d62728",
        "CGT-v1": "#9467bd",
        "CGT-v2": "#2ca02c",
        "CGT-v3": "#006400"
    }

    win_smooth = 50
    for title, x_min, x_max, ax, s_on, s_off in plot_windows:
        ax.axvspan(s_on, s_off, color="#ffdddd", alpha=0.5, label="Stress Shock Window")

        for p_name in ["Always-HT", "Always-ARF", "CGT-v1", "CGT-v2", "CGT-v3"]:
            corr = sim_results[p_name]["correct"]
            smooth_acc = pd.Series(corr).rolling(win_smooth, min_periods=10).mean() * 100.0
            ax.plot(range(x_min, x_max), smooth_acc[x_min:x_max], label=p_name,
                    color=colors[p_name], linewidth=2.0 if "CGT-v3" in p_name else 1.5,
                    linestyle="-" if "CGT" in p_name else "--")

        # Mark ADWIN alarms for CGT-v3
        v3_alarms = [a for a in sim_results["CGT-v3"]["alarms"] if x_min <= a <= x_max]
        for a in v3_alarms:
            ax.axvline(a, color="red", linestyle=":", linewidth=1.5, alpha=0.8, label="ADWIN Alarm (CGT-v3)" if a == v3_alarms[0] else "")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Sliding Accuracy (W=50, %)", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(45, 100)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=8, ncol=3)

    axes[2].set_xlabel("Test Stream Instance Index (Hours)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_fig = "results/figures/fig_day7_drift_robustness.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Generated drift robustness figure: {out_fig}", flush=True)

if __name__ == "__main__":
    run_drift_robustness()

