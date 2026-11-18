"""
Task D: Comprehensive Performance Metrics Beyond Accuracy & Temporal Disaggregation.
Evaluates:
  1. Balanced accuracy: (Recall_0 + Recall_1) / 2
  2. Per-class Precision, Recall, F1 for Class 0 and Class 1
  3. Cohen's Kappa (kappa)
  4. Temporal Kappa vs Persistence (kappa_temporal = (p_o - p_p) / (1 - p_p), Bifet kappa+)
  5. Per-month and per-season temporal disaggregation for HT, ARF, and CGT-v3
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def compute_extended_metrics():
    print("=================================================================", flush=True)
    print("TASK D: EXTENDED METRICS BEYOND ACCURACY & TEMPORAL DISAGGREGATION", flush=True)
    print("=================================================================\n", flush=True)

    # Load timestamps from clean dataset
    data_df = pd.read_csv("data/energy_dataset_clean.csv")
    data_df['Parsed_Timestamp'] = pd.to_datetime(data_df['Timestamp'], format='mixed')
    data_df = data_df.sort_values('Parsed_Timestamp').reset_index(drop=True)

    # Test slice: after 24-hr lag warmup and 10,512 validation instances
    val_len = 10512
    test_timestamps = data_df['Parsed_Timestamp'].values[24 + val_len:]
    N = len(test_timestamps)

    model_files = {
        "Persistence (1-Hr Momentum)": "results/raw/step_predictions_persistence_1-hr_momentum.csv",
        "Online SGD (Logistic Regression)": "results/raw/step_predictions_online_sgd_logistic_regression.csv",
        "Always-HT": "results/raw/step_predictions_always-ht.csv",
        "A3: Optimized ARF": "results/raw/step_predictions_a3_optimized_arf.csv",
        "Always-ARF": "results/raw/step_predictions_always-arf.csv",
        "OMS-MAB (Wilson et al., 2024)": "results/raw/step_predictions_oms-mab_wilson_et_al._2024.csv",
        "CGT (Proposed, S1)": "results/raw/step_predictions_cgt_proposed_s1.csv",
        "CGT-v2 (Adaptive Dual-Gate)": "results/raw/step_predictions_cgt-v2_adaptive_dual-gate.csv",
        "CGT-v3 (Learned Meta-Gate)": "results/raw/step_predictions_cgt-v3_learned_meta-gate.csv"
    }

    # First load persistence accuracy p_p
    p_df = pd.read_csv(model_files["Persistence (1-Hr Momentum)"])
    p_persistence = (p_df["y_true"] == p_df["y_pred"]).mean()
    print(f"Persistence Reference Accuracy (p_p): {p_persistence*100:.2f}%\n", flush=True)

    metrics_rows = []
    loaded_preds = {}

    for name, fpath in model_files.items():
        if not os.path.exists(fpath):
            print(f"Warning: Missing {fpath}", flush=True)
            continue
        df = pd.read_csv(fpath)
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        loaded_preds[name] = y_pred

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        acc = (tp + tn) / N
        rec0 = tn / (tn + fp)
        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

        rec1 = tp / (tp + fn)
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_1 = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0

        balanced_acc = (rec0 + rec1) / 2.0
        kappa = cohen_kappa_score(y_true, y_pred)
        kappa_temporal = (acc - p_persistence) / (1.0 - p_persistence)

        metrics_rows.append({
            "model_name": name,
            "accuracy_pct": round(acc * 100.0, 2),
            "balanced_accuracy_pct": round(balanced_acc * 100.0, 2),
            "class0_precision_pct": round(prec0 * 100.0, 2),
            "class0_recall_pct": round(rec0 * 100.0, 2),
            "class0_f1": round(f1_0, 4),
            "class1_precision_pct": round(prec1 * 100.0, 2),
            "class1_recall_pct": round(rec1 * 100.0, 2),
            "class1_f1": round(f1_1, 4),
            "cohen_kappa": round(kappa, 4),
            "kappa_temporal_bifet": round(kappa_temporal, 4)
        })

    ext_df = pd.DataFrame(metrics_rows)
    print("--- Performance Metrics Beyond Accuracy ---")
    print(ext_df[["model_name", "accuracy_pct", "balanced_accuracy_pct", "class0_f1", "class1_f1", "cohen_kappa", "kappa_temporal_bifet"]].to_string(index=False))

    ext_df.to_csv("results/tables/day7_extended_metrics.csv", index=False)
    print("\nSaved extended metrics to results/tables/day7_extended_metrics.csv\n", flush=True)

    # 2. Temporal Disaggregation for HT, ARF, and CGT-v3
    y_true_canonical = pd.read_csv(model_files["Always-HT"])["y_true"].values
    time_series_df = pd.DataFrame({
        "timestamp": pd.to_datetime(test_timestamps),
        "y_true": y_true_canonical,
        "pred_ht": loaded_preds["Always-HT"],
        "pred_arf": loaded_preds["Always-ARF"],
        "pred_cgt_v3": loaded_preds["CGT-v3 (Learned Meta-Gate)"]
    })

    time_series_df["correct_ht"] = (time_series_df["pred_ht"] == time_series_df["y_true"]).astype(int)
    time_series_df["correct_arf"] = (time_series_df["pred_arf"] == time_series_df["y_true"]).astype(int)
    time_series_df["correct_cgt_v3"] = (time_series_df["pred_cgt_v3"] == time_series_df["y_true"]).astype(int)

    time_series_df["year_month"] = time_series_df["timestamp"].dt.to_period("M").astype(str)
    time_series_df["season"] = time_series_df["timestamp"].dt.month.map(get_season)

    # Monthly breakdown
    monthly_summary = time_series_df.groupby("year_month").agg(
        n_instances=("y_true", "count"),
        acc_ht=("correct_ht", lambda x: round(x.mean() * 100.0, 2)),
        acc_arf=("correct_arf", lambda x: round(x.mean() * 100.0, 2)),
        acc_cgt_v3=("correct_cgt_v3", lambda x: round(x.mean() * 100.0, 2))
    ).reset_index()
    monthly_summary["delta_cgt_minus_ht"] = (monthly_summary["acc_cgt_v3"] - monthly_summary["acc_ht"]).round(2)
    monthly_summary["delta_arf_minus_ht"] = (monthly_summary["acc_arf"] - monthly_summary["acc_ht"]).round(2)

    # Seasonal breakdown
    seasonal_summary = time_series_df.groupby("season").agg(
        n_instances=("y_true", "count"),
        acc_ht=("correct_ht", lambda x: round(x.mean() * 100.0, 2)),
        acc_arf=("correct_arf", lambda x: round(x.mean() * 100.0, 2)),
        acc_cgt_v3=("correct_cgt_v3", lambda x: round(x.mean() * 100.0, 2))
    ).reset_index()
    seasonal_summary["delta_cgt_minus_ht"] = (seasonal_summary["acc_cgt_v3"] - seasonal_summary["acc_ht"]).round(2)
    seasonal_summary["delta_arf_minus_ht"] = (seasonal_summary["acc_arf"] - seasonal_summary["acc_ht"]).round(2)

    print("--- Seasonal Disaggregation (HT vs ARF vs CGT-v3) ---")
    print(seasonal_summary.to_string(index=False))

    print("\n--- Monthly Disaggregation Summary (First 6 and Last 6 Months) ---")
    print(pd.concat([monthly_summary.head(6), monthly_summary.tail(6)]).to_string(index=False))

    monthly_summary.to_csv("results/tables/day7_monthly_disaggregation.csv", index=False)
    seasonal_summary.to_csv("results/tables/day7_seasonal_disaggregation.csv", index=False)
    print("\nSaved monthly and seasonal tables to results/tables/", flush=True)

if __name__ == "__main__":
    compute_extended_metrics()

