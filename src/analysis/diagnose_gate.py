import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper, ARFOptimizedWrapper
from src.controllers.cgt import ConditionalGatedTrainingController

def run_gate_diagnosis():
    print("=================================================================")
    print("DAY 2: GATE DIAGNOSIS & DORMANT VS AWAKE PERFORMANCE ANALYSIS")
    print("=================================================================\n")

    # Load frozen config
    with open("configs/frozen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    loader = ElectricityStreamLoader(cfg["data"]["csv_path"])
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_len = cfg["data"]["val_instances"]
    test_stream = full_stream[val_len:]
    n_test = len(test_stream)

    print(f"Test stream length: {n_test} instances")

    # Instantiate models
    ht = HoeffdingTreeWrapper()
    arf = ARFFullWrapper(seed=cfg["project"]["seed"])
    cgt = ConditionalGatedTrainingController(
        stability_k=cfg["cgt_frozen"]["stability_k"],
        buffer_b=cfg["cgt_frozen"]["buffer_b"],
        warm_start_strategy="S1",
        adwin_delta=cfg["cgt_frozen"]["adwin_delta"],
        adwin_clock=cfg["cgt_frozen"]["adwin_clock"],
        seed=cfg["project"]["seed"]
    )
    cgt_plus = ConditionalGatedTrainingController(
        stability_k=1000,
        buffer_b=300,
        warm_start_strategy="S1",
        adwin_delta=0.02,
        adwin_clock=16,
        use_pruned_arf=True,
        seed=cfg["project"]["seed"]
    )

    # Tracking lists
    steps = []
    timestamps = []
    y_trues = []
    
    preds_ht = []
    preds_arf = []
    preds_cgt = []
    preds_cgt_plus = []
    
    gate_cgt = []
    gate_cgt_plus = []
    alarm_cgt = []
    alarm_cgt_plus = []

    # Stream processing loop
    for t_idx, (x_t, y_t, ts) in enumerate(test_stream):
        steps.append(t_idx + 1)
        timestamps.append(ts)
        y_trues.append(y_t)

        # Predictions before label reveal
        p_ht = ht.predict_one(x_t)
        p_arf = arf.predict_one(x_t)
        
        g_cgt = 1 if cgt.gate_open else 0
        p_cgt = cgt.predict_one(x_t)
        
        g_cgt_plus = 1 if cgt_plus.gate_open else 0
        p_cgt_plus = cgt_plus.predict_one(x_t)

        preds_ht.append(p_ht)
        preds_arf.append(p_arf)
        preds_cgt.append(p_cgt)
        preds_cgt_plus.append(p_cgt_plus)
        
        gate_cgt.append(g_cgt)
        gate_cgt_plus.append(g_cgt_plus)

        # Updates / Training
        ht.learn_one(x_t, y_t)
        arf.learn_one(x_t, y_t)
        
        cgt.learn_one(x_t, y_t, p_cgt)
        alarm_cgt.append(1 if cgt.detector.drift_detected else 0)
        
        cgt_plus.learn_one(x_t, y_t, p_cgt_plus)
        alarm_cgt_plus.append(1 if cgt_plus.detector.drift_detected else 0)

    # Convert to numpy arrays for vector analysis
    y_arr = np.array(y_trues)
    p_ht_arr = np.array(preds_ht)
    p_arf_arr = np.array(preds_arf)
    p_cgt_arr = np.array(preds_cgt)
    p_cgt_plus_arr = np.array(preds_cgt_plus)
    g_cgt_arr = np.array(gate_cgt)
    g_cgt_plus_arr = np.array(gate_cgt_plus)
    a_cgt_arr = np.array(alarm_cgt)
    a_cgt_plus_arr = np.array(alarm_cgt_plus)

    # Accuracy calculations
    correct_ht = (p_ht_arr == y_arr)
    correct_arf = (p_arf_arr == y_arr)
    correct_cgt = (p_cgt_arr == y_arr)
    correct_cgt_plus = (p_cgt_plus_arr == y_arr)

    # Breakdown for CGT (S1)
    dormant_mask = (g_cgt_arr == 0)
    awake_mask = (g_cgt_arr == 1)

    n_dormant = np.sum(dormant_mask)
    n_awake = np.sum(awake_mask)
    frac_dormant = n_dormant / n_test
    frac_awake = n_awake / n_test

    acc_ht_all = np.mean(correct_ht) * 100.0
    acc_arf_all = np.mean(correct_arf) * 100.0
    acc_cgt_all = np.mean(correct_cgt) * 100.0

    acc_ht_dormant = np.mean(correct_ht[dormant_mask]) * 100.0
    acc_arf_dormant = np.mean(correct_arf[dormant_mask]) * 100.0
    acc_cgt_dormant = np.mean(correct_cgt[dormant_mask]) * 100.0

    acc_ht_awake = np.mean(correct_ht[awake_mask]) * 100.0 if n_awake > 0 else 0.0
    acc_arf_awake = np.mean(correct_arf[awake_mask]) * 100.0 if n_awake > 0 else 0.0
    acc_cgt_awake = np.mean(correct_cgt[awake_mask]) * 100.0 if n_awake > 0 else 0.0

    # Breakdown for CGT-Plus
    dormant_plus_mask = (g_cgt_plus_arr == 0)
    awake_plus_mask = (g_cgt_plus_arr == 1)
    n_dormant_plus = np.sum(dormant_plus_mask)
    n_awake_plus = np.sum(awake_plus_mask)

    acc_cgt_plus_dormant = np.mean(correct_cgt_plus[dormant_plus_mask]) * 100.0
    acc_cgt_plus_awake = np.mean(correct_cgt_plus[awake_plus_mask]) * 100.0

    # Contingency matrix between HT and ARF:
    both_correct = np.sum(correct_ht & correct_arf)
    ht_only_correct = np.sum(correct_ht & (~correct_arf))
    arf_only_correct = np.sum((~correct_ht) & correct_arf)
    neither_correct = np.sum((~correct_ht) & (~correct_arf))

    print("\n--------------------------------------------------------------")
    print("CGT (S1) DORMANT VS AWAKE BREAKDOWN:")
    print("--------------------------------------------------------------")
    print(f"Total Test Instances: {n_test}")
    print(f"Dormant Time: {n_dormant} hours ({frac_dormant*100:.2f}%)")
    print(f"Awake Time:   {n_awake} hours ({frac_awake*100:.2f}%)")
    print(f"Total ADWIN Alarms: {np.sum(a_cgt_arr)}")
    print(f"\nOverall Accuracies: HT={acc_ht_all:.2f}%, ARF={acc_arf_all:.2f}%, CGT={acc_cgt_all:.2f}%")
    print(f"\nDormant Period Accuracies (g_t = 0, {n_dormant} inst):")
    print(f"  Always-HT:  {acc_ht_dormant:.2f}%")
    print(f"  Always-ARF: {acc_arf_dormant:.2f}%  <-- ARF is +{acc_arf_dormant - acc_ht_dormant:.2f}% better even while asleep!")
    print(f"  CGT:        {acc_cgt_dormant:.2f}%  (Matches HT exactly because HT is queried)")
    print(f"\nAwake Period Accuracies (g_t = 1, {n_awake} inst):")
    print(f"  Always-HT:  {acc_ht_awake:.2f}%")
    print(f"  Always-ARF: {acc_arf_awake:.2f}%")
    print(f"  CGT:        {acc_cgt_awake:.2f}%  (Matches ARF adaptation during drift)")

    print("\n--------------------------------------------------------------")
    print("CGT-PLUS DORMANT VS AWAKE BREAKDOWN:")
    print("--------------------------------------------------------------")
    print(f"Dormant Time: {n_dormant_plus} hours ({n_dormant_plus/n_test*100:.2f}%)")
    print(f"Awake Time:   {n_awake_plus} hours ({n_awake_plus/n_test*100:.2f}%)")
    print(f"Total ADWIN Alarms: {np.sum(a_cgt_plus_arr)}")
    print(f"Dormant Accuracy: {acc_cgt_plus_dormant:.2f}%")
    print(f"Awake Accuracy:   {acc_cgt_plus_awake:.2f}%")

    print("\n--------------------------------------------------------------")
    print("HT VS ARF CONTINGENCY MATRIX (OVERALL STREAM):")
    print("--------------------------------------------------------------")
    print(f"Both Correct (HT=1, ARF=1):       {both_correct} ({both_correct/n_test*100:.2f}%) -> Cheaper HT is sufficient")
    print(f"ARF-Only Correct (HT=0, ARF=1):   {arf_only_correct} ({arf_only_correct/n_test*100:.2f}%) -> Gate MUST awaken ARF here")
    print(f"HT-Only Correct (HT=1, ARF=0):    {ht_only_correct} ({ht_only_correct/n_test*100:.2f}%) -> HT beats ARF")
    print(f"Neither Correct (HT=0, ARF=0):    {neither_correct} ({neither_correct/n_test*100:.2f}%) -> Irreducible error for both")

    # Save detailed breakdown table
    df_breakdown = pd.DataFrame([
        {
            "Controller": "CGT (Proposed, S1)",
            "Dormant_Hours": n_dormant,
            "Dormant_Pct": round(frac_dormant * 100, 2),
            "Awake_Hours": n_awake,
            "Awake_Pct": round(frac_awake * 100, 2),
            "ADWIN_Alarms": int(np.sum(a_cgt_arr)),
            "Dormant_Acc_CGT": round(acc_cgt_dormant, 2),
            "Dormant_Acc_HT": round(acc_ht_dormant, 2),
            "Dormant_Acc_ARF": round(acc_arf_dormant, 2),
            "Awake_Acc_CGT": round(acc_cgt_awake, 2),
            "Awake_Acc_HT": round(acc_ht_awake, 2),
            "Awake_Acc_ARF": round(acc_arf_awake, 2),
            "Overall_Acc_CGT": round(acc_cgt_all, 2)
        },
        {
            "Controller": "CGT-Plus (Optimized Hybrid)",
            "Dormant_Hours": n_dormant_plus,
            "Dormant_Pct": round(n_dormant_plus / n_test * 100, 2),
            "Awake_Hours": n_awake_plus,
            "Awake_Pct": round(n_awake_plus / n_test * 100, 2),
            "ADWIN_Alarms": int(np.sum(a_cgt_plus_arr)),
            "Dormant_Acc_CGT": round(acc_cgt_plus_dormant, 2),
            "Dormant_Acc_HT": round(acc_ht_dormant, 2),
            "Dormant_Acc_ARF": round(acc_arf_dormant, 2),
            "Awake_Acc_CGT": round(acc_cgt_plus_awake, 2),
            "Awake_Acc_HT": round(acc_ht_awake, 2),
            "Awake_Acc_ARF": round(acc_arf_awake, 2),
            "Overall_Acc_CGT": round(np.mean(correct_cgt_plus) * 100.0, 2)
        }
    ])
    os.makedirs("results/tables", exist_ok=True)
    df_breakdown.to_csv("results/tables/cgt_dormant_vs_awake_breakdown.csv", index=False)
    print("\nSaved breakdown to results/tables/cgt_dormant_vs_awake_breakdown.csv")

    # Contingency summary table
    df_contingency = pd.DataFrame([
        {"State": "Both Correct (HT=1, ARF=1)", "Count": both_correct, "Percentage": round(both_correct/n_test*100, 2), "Optimal Arm": "HT (Lowest Cost)"},
        {"State": "ARF-Only Correct (HT=0, ARF=1)", "Count": arf_only_correct, "Percentage": round(arf_only_correct/n_test*100, 2), "Optimal Arm": "ARF (Essential Win)"},
        {"State": "HT-Only Correct (HT=1, ARF=0)", "Count": ht_only_correct, "Percentage": round(ht_only_correct/n_test*100, 2), "Optimal Arm": "HT (Accuracy + Cost)"},
        {"State": "Neither Correct (HT=0, ARF=0)", "Count": neither_correct, "Percentage": round(neither_correct/n_test*100, 2), "Optimal Arm": "HT (Avoid Wasted ARF Cost)"}
    ])
    df_contingency.to_csv("results/tables/ht_vs_arf_contingency.csv", index=False)
    print("Saved contingency analysis to results/tables/ht_vs_arf_contingency.csv")

    # Compute sliding window error and accuracy for plotting (window = 500)
    w = 500
    sliding_acc_ht = pd.Series(correct_ht.astype(float)).rolling(w, min_periods=50).mean() * 100.0
    sliding_acc_arf = pd.Series(correct_arf.astype(float)).rolling(w, min_periods=50).mean() * 100.0
    sliding_acc_cgt = pd.Series(correct_cgt.astype(float)).rolling(w, min_periods=50).mean() * 100.0
    sliding_err_diff = (pd.Series((~correct_ht).astype(float)) - pd.Series((~correct_arf).astype(float))).rolling(w, min_periods=50).mean() * 100.0

    # Generate Diagnostic Figure: Gate State vs ADWIN Alarms vs Error Dynamics
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1.2]})

    # Panel 1: Sliding Accuracy
    axes[0].plot(steps, sliding_acc_arf, label="Always-ARF (Ensemble)", color="#2ca02c", linewidth=1.8, alpha=0.9)
    axes[0].plot(steps, sliding_acc_ht, label="Always-HT (Single Tree)", color="#1f77b4", linewidth=1.5, alpha=0.8)
    axes[0].plot(steps, sliding_acc_cgt, label="CGT (Gated Switch)", color="#d62728", linewidth=2.0, linestyle="--")
    
    # Mark ADWIN Alarms
    alarm_indices = np.where(a_cgt_arr == 1)[0]
    for idx in alarm_indices:
        axes[0].axvline(x=steps[idx], color="black", linestyle=":", alpha=0.7, label="ADWIN Alarm" if idx == alarm_indices[0] else "")

    axes[0].set_ylabel("Prequential Acc (%) [W=500]", fontsize=11)
    axes[0].set_title("Day 2 Diagnostic: Gate State vs. ADWIN Alarms vs. Tracking Error", fontsize=13, fontweight='bold')
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower left", frameon=True, fontsize=10)
    axes[0].set_ylim(75, 96)

    # Panel 2: Gate State g_t
    axes[1].fill_between(steps, 0, g_cgt_arr, color="#d62728", alpha=0.35, step="pre", label="CGT Gate State (1=Awake, 0=Dormant)")
    axes[1].plot(steps, g_cgt_arr, color="#d62728", linewidth=1.2, drawstyle="steps-pre")
    axes[1].set_ylabel("Gate $g_t$", fontsize=11)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["0 (Dormant)", "1 (Awake)"])
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", frameon=True, fontsize=10)

    # Panel 3: Error Difference (HT Error - ARF Error)
    axes[2].plot(steps, sliding_err_diff, color="#9467bd", linewidth=1.6, label="Sliding Error Gap: (HT Error - ARF Error) %")
    axes[2].axhline(y=0, color="gray", linestyle="-", alpha=0.6)
    axes[2].fill_between(steps, 0, sliding_err_diff, where=(sliding_err_diff > 0), color="#2ca02c", alpha=0.25, label="ARF Superior (Gap > 0)")
    axes[2].fill_between(steps, 0, sliding_err_diff, where=(sliding_err_diff < 0), color="#1f77b4", alpha=0.25, label="HT Superior (Gap < 0)")
    axes[2].set_ylabel("Error Gap (%)", fontsize=11)
    axes[2].set_xlabel("Test Stream Instance (Hours, mid 2016 – end 2018)", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    diag_fig_path = "results/figures/fig_day2_gate_diagnosis.png"
    plt.savefig(diag_fig_path, dpi=300)
    plt.close()
    print(f"Generated diagnostic figure: {diag_fig_path}")

    return df_breakdown, df_contingency

if __name__ == "__main__":
    run_gate_diagnosis()

