"""
Task A: Diagnostic Analysis of CGT-v3 Meta-Gate Skill and Gap Closure.
Evaluates:
  1. Base rate of z_t = I(ARF correct and HT wrong).
  2. Confusion matrix of binary gate decision g_t vs z_t (TP, FP, FN, TN).
  3. Gate Precision, Recall, Specificity, F1.
  4. ROC-AUC and PR-AUC of continuous meta-model score P(z_t=1|s_t) vs z_t.
  5. Benchmark comparison vs:
     a) Random gate with identical duty cycle rho.
     b) Heuristic gate using only HT margin M(x).
  6. Breakdown of accuracy gained vs lost across all gate activations:
     - ARF-right / HT-wrong (Net Gain: +1)
     - ARF-wrong / HT-right (Net Harm: -1)
     - Both right (Neutral: 0)
     - Both wrong (Neutral: 0)
"""

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from src.data.stream_loader import ElectricityStreamLoader
from src.controllers.cgt_v3 import CGTv3LearnedGateController
from src.models.base_models import HoeffdingTreeWrapper, ARFOptimizedWrapper

def run_gate_meta_eval():
    with open("configs/frozen_v3.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    v3_params = cfg.get("cgt_v3_frozen", {})
    tau_gate = v3_params.get("tau_gate", 0.20)
    max_rho = v3_params.get("max_rho", 0.10)
    h_cooldown = v3_params.get("h_cooldown", 50)
    buffer_b = v3_params.get("buffer_b", 200)
    recent_w = v3_params.get("recent_w", 50)
    use_pruned_arf = v3_params.get("use_pruned_arf", True)
    seed = cfg.get("project", {}).get("seed", 42)

    controller = CGTv3LearnedGateController(
        tau_gate=tau_gate,
        max_rho=max_rho,
        h_cooldown=h_cooldown,
        buffer_b=buffer_b,
        recent_w=recent_w,
        use_pruned_arf=use_pruned_arf,
        seed=seed
    )

    loader = ElectricityStreamLoader(csv_path="data/energy_dataset_clean.csv")
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_len = cfg.get("data", {}).get("val_instances", 10512)
    val_stream = full_stream[:val_len]
    test_stream = full_stream[val_len:]

    print(f"Warming up controller on {len(val_stream)} validation instances...", flush=True)
    for x, y, ts in val_stream:
        y_pred = controller.predict_one(x)
        controller.learn_one(x, y, y_pred)

    print(f"Evaluating diagnostic gate signals on {len(test_stream)} test instances...", flush=True)

    records = []
    rng = np.random.RandomState(seed)

    for idx, (x, y, ts) in enumerate(test_stream, start=1):
        # 1. Inspect HT predictive margin
        ht_proba = controller.ht.model.predict_proba_one(x)
        p0 = ht_proba.get(0, 0.5)
        p1 = ht_proba.get(1, 0.5)
        ht_margin = abs(p1 - p0)

        # 2. Extract meta-features and continuous score
        meta_feat = controller._extract_meta_features(x)
        proba_meta = controller.meta_gate.predict_proba_one(meta_feat)
        meta_score = proba_meta.get(1, 0.0)

        # 3. Base learner individual predictions
        pred_ht = controller.ht.predict_one(x)
        pred_arf = controller.arf.predict_one(x)

        # 4. Controller gate decision and final prediction
        y_pred = controller.predict_one(x)
        gate_open = 1 if controller.gate_open else 0

        # 5. Calculate true target z_t
        ht_correct = (pred_ht == y)
        arf_correct = (pred_arf == y)
        z_t = 1 if (arf_correct and not ht_correct) else 0

        # Learn step
        controller.learn_one(x, y, y_pred)

        records.append({
            "idx": idx,
            "y_true": y,
            "pred_ht": pred_ht,
            "pred_arf": pred_arf,
            "y_pred": y_pred,
            "gate_open": gate_open,
            "meta_score": meta_score,
            "ht_margin": ht_margin,
            "z_t": z_t,
            "ht_correct": ht_correct,
            "arf_correct": arf_correct
        })

    df = pd.DataFrame(records)
    N = len(df)

    # 1. Base rate of z_t
    base_rate_z = df["z_t"].mean()
    total_z1 = df["z_t"].sum()

    # 2. Gate confusion matrix vs z_t
    tn, fp, fn, tp = confusion_matrix(df["z_t"], df["gate_open"]).ravel()
    gate_rho = df["gate_open"].mean()
    gate_total_open = df["gate_open"].sum()

    precision_gate = tp / max(tp + fp, 1)
    recall_gate = tp / max(tp + fn, 1)
    specificity_gate = tn / max(tn + fp, 1)
    f1_gate = 2 * precision_gate * recall_gate / max(precision_gate + recall_gate, 1e-9)

    # 3. ROC-AUC and PR-AUC of meta_score
    roc_auc_meta = roc_auc_score(df["z_t"], df["meta_score"])
    pr_auc_meta = average_precision_score(df["z_t"], df["meta_score"])

    # 4. Heuristic Gate: HT margin (Low margin => higher uncertainty)
    # Using (1 - margin) as score for predicting z_t
    uncertainty_score = 1.0 - df["ht_margin"]
    roc_auc_margin = roc_auc_score(df["z_t"], uncertainty_score)
    pr_auc_margin = average_precision_score(df["z_t"], uncertainty_score)

    # Margin thresholded gate with identical duty cycle
    margin_threshold = np.quantile(df["ht_margin"], gate_rho)
    gate_margin = (df["ht_margin"] <= margin_threshold).astype(int)
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(df["z_t"], gate_margin).ravel()
    precision_margin = tp_m / max(tp_m + fp_m, 1)
    recall_margin = tp_m / max(tp_m + fn_m, 1)
    f1_margin = 2 * precision_margin * recall_margin / max(precision_margin + recall_margin, 1e-9)

    # 5. Random Gate with identical duty cycle rho
    gate_rand = (rng.rand(N) < gate_rho).astype(int)
    tn_r, fp_r, fn_r, tp_r = confusion_matrix(df["z_t"], gate_rand).ravel()
    precision_rand = tp_r / max(tp_r + fp_r, 1)
    recall_rand = tp_r / max(tp_r + fn_r, 1)
    f1_rand = 2 * precision_rand * recall_rand / max(precision_rand + recall_rand, 1e-9)

    # 6. Accuracy Gained vs Lost breakdown when gate is active (gate_open == 1)
    active_df = df[df["gate_open"] == 1]
    n_active = len(active_df)

    cat_arf_right_ht_wrong = ((active_df["pred_arf"] == active_df["y_true"]) & (active_df["pred_ht"] != active_df["y_true"])).sum()
    cat_arf_wrong_ht_right = ((active_df["pred_arf"] != active_df["y_true"]) & (active_df["pred_ht"] == active_df["y_true"])).sum()
    cat_both_right = ((active_df["pred_arf"] == active_df["y_true"]) & (active_df["pred_ht"] == active_df["y_true"])).sum()
    cat_both_wrong = ((active_df["pred_arf"] != active_df["y_true"]) & (active_df["pred_ht"] != active_df["y_true"])).sum()

    net_correct_gain = cat_arf_right_ht_wrong - cat_arf_wrong_ht_right
    net_acc_delta_pct = (net_correct_gain / N) * 100.0

    print("\n" + "="*70)
    print("TASK A: CGT-v3 GATE META-EVALUATION ON TEST STREAM (24,528 instances)")
    print("="*70)
    print(f"Base Rate of z_t = I(ARF right and HT wrong): {base_rate_z*100:.2f}% ({total_z1}/{N} instances)")
    print(f"Gate Active Steps (rho): {gate_rho*100:.2f}% ({gate_total_open}/{N} instances)")
    print("\n--- Gate Decision vs z_t Confusion Matrix ---")
    print(f"TP (Gate=1, z=1): {tp:>5}  |  FP (Gate=1, z=0): {fp:>5}")
    print(f"FN (Gate=0, z=1): {fn:>5}  |  TN (Gate=0, z=0): {tn:>5}")
    print(f"Gate Precision: {precision_gate*100:.2f}% (vs Random {base_rate_z*100:.2f}%)")
    print(f"Gate Recall:    {recall_gate*100:.2f}%")
    print(f"Gate Specificity: {specificity_gate*100:.2f}%")
    print(f"Gate F1-Score:  {f1_gate:.4f}")
    print(f"Meta-Model Score ROC-AUC: {roc_auc_meta:.4f}")
    print(f"Meta-Model Score PR-AUC:  {pr_auc_meta:.4f} (Base rate: {base_rate_z:.4f})")

    print("\n--- Gate Comparison against Alternatives (Same Duty Cycle rho) ---")
    print(f"1. Random Gate (rho={gate_rho*100:.2f}%):")
    print(f"   Precision: {precision_rand*100:.2f}%, Recall: {recall_rand*100:.2f}%, F1: {f1_rand:.4f}")
    print(f"2. Heuristic Gate (HT Margin M(x)):")
    print(f"   Precision: {precision_margin*100:.2f}%, Recall: {recall_margin*100:.2f}%, F1: {f1_margin:.4f}")
    print(f"   ROC-AUC: {roc_auc_margin:.4f}, PR-AUC: {pr_auc_margin:.4f}")
    print(f"3. CGT-v3 Learned Meta-Gate:")
    print(f"   Precision: {precision_gate*100:.2f}%, Recall: {recall_gate*100:.2f}%, F1: {f1_gate:.4f}")
    print(f"   ROC-AUC: {roc_auc_meta:.4f}, PR-AUC: {pr_auc_meta:.4f}")

    print("\n--- Breakdown of Accuracy Gained vs Lost Across Gate Activations ---")
    print(f"Total Gate Activations: {n_active} instances ({n_active/N*100:.2f}%)")
    print(f"  [+] ARF Right, HT Wrong (True Gain):  {cat_arf_right_ht_wrong:>5} ({cat_arf_right_ht_wrong/n_active*100:.2f}%) -> +{cat_arf_right_ht_wrong} correct predictions")
    print(f"  [-] ARF Wrong, HT Right (Harm/Loss):  {cat_arf_wrong_ht_right:>5} ({cat_arf_wrong_ht_right/n_active*100:.2f}%) -> -{cat_arf_wrong_ht_right} correct predictions")
    print(f"  [=] Both Models Right (Neutral):      {cat_both_right:>5} ({cat_both_right/n_active*100:.2f}%)")
    print(f"  [=] Both Models Wrong (Neutral):      {cat_both_wrong:>5} ({cat_both_wrong/n_active*100:.2f}%)")
    print(f"Net Correct Gain from Gate: {net_correct_gain:+d} predictions")
    print(f"Net Accuracy Contribution to Stream: {net_acc_delta_pct:+.2f}%")
    print("="*70)

    # Save summary table
    summary_df = pd.DataFrame([{
        "metric": "Base Rate of z_t", "value": f"{base_rate_z*100:.2f}%"
    }, {
        "metric": "Gate Duty Cycle (rho)", "value": f"{gate_rho*100:.2f}%"
    }, {
        "metric": "Gate True Positives (TP)", "value": tp
    }, {
        "metric": "Gate False Positives (FP)", "value": fp
    }, {
        "metric": "Gate False Negatives (FN)", "value": fn
    }, {
        "metric": "Gate True Negatives (TN)", "value": tn
    }, {
        "metric": "Gate Precision", "value": f"{precision_gate*100:.2f}%"
    }, {
        "metric": "Gate Recall", "value": f"{recall_gate*100:.2f}%"
    }, {
        "metric": "Gate Specificity", "value": f"{specificity_gate*100:.2f}%"
    }, {
        "metric": "Gate F1 Score", "value": f"{f1_gate:.4f}"
    }, {
        "metric": "Meta-Score ROC-AUC", "value": f"{roc_auc_meta:.4f}"
    }, {
        "metric": "Meta-Score PR-AUC", "value": f"{pr_auc_meta:.4f}"
    }, {
        "metric": "Margin Gate Precision", "value": f"{precision_margin*100:.2f}%"
    }, {
        "metric": "Margin Gate Recall", "value": f"{recall_margin*100:.2f}%"
    }, {
        "metric": "Margin Gate ROC-AUC", "value": f"{roc_auc_margin:.4f}"
    }, {
        "metric": "Margin Gate PR-AUC", "value": f"{pr_auc_margin:.4f}"
    }, {
        "metric": "Random Gate Precision", "value": f"{precision_rand*100:.2f}%"
    }, {
        "metric": "Random Gate Recall", "value": f"{recall_rand*100:.2f}%"
    }, {
        "metric": "Gate ARF-right / HT-wrong (Gain)", "value": cat_arf_right_ht_wrong
    }, {
        "metric": "Gate ARF-wrong / HT-right (Loss)", "value": cat_arf_wrong_ht_right
    }, {
        "metric": "Gate Both-Right (Neutral)", "value": cat_both_right
    }, {
        "metric": "Gate Both-Wrong (Neutral)", "value": cat_both_wrong
    }, {
        "metric": "Net Correct Predictions Gained", "value": net_correct_gain
    }, {
        "metric": "Net Stream Accuracy Delta", "value": f"{net_acc_delta_pct:+.2f}%"
    }])

    os.makedirs("results/tables", exist_ok=True)
    summary_df.to_csv("results/tables/day7_gate_meta_eval.csv", index=False)
    print("Saved Task A diagnostic table to results/tables/day7_gate_meta_eval.csv", flush=True)

if __name__ == "__main__":
    run_gate_meta_eval()
