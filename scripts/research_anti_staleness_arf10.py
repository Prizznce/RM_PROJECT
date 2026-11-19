"""
Anti-Staleness ARF-10 Exploration Harness
=============================================================================
Investigates:
1. Theoretical Oracle Upper Bound for Cascaded Selection (HT / SGD -> ARF-10)
   as a function of duty budget rho in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30].
2. The Concept Staleness Gap:
   - Full-Stream Trained ARF-10 vs Gated-Only Trained ARF-10 vs Subsampled Background ARF-10.
3. Anti-Staleness CGT-v5 Controller Prototype:
   - Evaluates on 5,000 test hours from Stream 1 and 5,000 test hours from Stream 2 (POSOCO).
"""

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier
from river.linear_model import LogisticRegression
from river.preprocessing import StandardScaler

from src.data.stream_loader import ElectricityStreamLoader
from scripts.run_real_posoco_benchmark import load_real_posoco_stream


def run_oracle_and_staleness_analysis(stream_data, stream_name="Stream 1", max_samples=5000):
    print(f"\n{'='*70}")
    print(f"ANALYSIS ON {stream_name} (First {max_samples} samples)")
    print(f"{'='*70}")
    
    samples = stream_data[:max_samples]
    
    # 1. Train HT, Linear SGD, and full Always-ARF-10 to collect ground-truth predictions
    ht = HoeffdingTreeClassifier(grace_period=50, delta=0.01)
    sgd = StandardScaler() | LogisticRegression()
    arf10_full = ARFClassifier(n_models=10, grace_period=50, lambda_value=6, delta=0.01, seed=42)
    
    ht_preds = []
    ht_margins = []
    sgd_preds = []
    arf10_full_preds = []
    y_trues = []
    
    print("Collecting step-by-step predictions for HT, SGD, and Always-ARF-10...")
    t0 = time.perf_counter()
    for item in samples:
        x, y = item[0], item[1]
        y_trues.append(y)
        
        # HT
        p_ht = ht.predict_proba_one(x)
        p0 = p_ht.get(0, 0.5)
        p1 = p_ht.get(1, 0.5)
        ht_pred = 1 if p1 > p0 else 0
        ht_margin = abs(p1 - p0)
        ht_preds.append(ht_pred)
        ht_margins.append(ht_margin)
        ht.learn_one(x, y)
        
        # SGD
        p_sgd = sgd.predict_proba_one(x)
        p0_s = p_sgd.get(0, 0.5)
        p1_s = p_sgd.get(1, 0.5)
        sgd_pred = 1 if p1_s > p0_s else 0
        sgd_preds.append(sgd_pred)
        sgd.learn_one(x, y)
        
        # ARF-10 (Full stream baseline)
        arf_pred = arf10_full.predict_one(x)
        arf10_full_preds.append(1 if arf_pred == 1 else 0)
        arf10_full.learn_one(x, y)
        
    t_elapsed = time.perf_counter() - t0
    y_trues = np.array(y_trues)
    ht_preds = np.array(ht_preds)
    ht_margins = np.array(ht_margins)
    sgd_preds = np.array(sgd_preds)
    arf10_full_preds = np.array(arf10_full_preds)
    
    acc_ht = np.mean(ht_preds == y_trues) * 100.0
    acc_sgd = np.mean(sgd_preds == y_trues) * 100.0
    acc_arf10 = np.mean(arf10_full_preds == y_trues) * 100.0
    
    print(f"Elapsed: {t_elapsed:.1f}s")
    print(f"Baseline Accuracies: HT = {acc_ht:.2f}%, SGD = {acc_sgd:.2f}%, Always-ARF-10 = {acc_arf10:.2f}%")
    
    # 2. Compute Oracle Bounds for different Fast Tiers and Duty Caps
    print("\n--- Theoretical Oracle Bounds (If Gate perfectly selects ARF-10 when Fast Tier is wrong) ---")
    for fast_name, f_preds in [("Hoeffding Tree", ht_preds), ("Linear SGD", sgd_preds)]:
        wrong_mask = (f_preds != y_trues)
        arf_saves = wrong_mask & (arf10_full_preds == y_trues)
        n_wrong = np.sum(wrong_mask)
        n_saves = np.sum(arf_saves)
        save_rate = n_saves / max(n_wrong, 1) * 100.0
        
        print(f"\nFast Tier: {fast_name}")
        print(f"  Total errors: {n_wrong} ({n_wrong/len(y_trues)*100:.1f}%)")
        print(f"  ARF-10 can fix: {n_saves} ({save_rate:.1f}% of errors fixed)")
        
        # What if duty cycle is capped at rho?
        for rho in [0.08, 0.12, 0.15, 0.20, 0.25]:
            max_gate_calls = int(rho * len(y_trues))
            # Oracle picks the instances where ARF fixes the error first
            fixes = min(n_saves, max_gate_calls)
            oracle_acc = (np.sum(f_preds == y_trues) + fixes) / len(y_trues) * 100.0
            print(f"  Oracle Acc at rho={rho*100:.0f}% (max {max_gate_calls} calls): {oracle_acc:.2f}%")

    # 3. Simulate The Concept Staleness Experiment:
    # Compare ARF-10 trained:
    # A) Only when HT margin < tau (Gate-only training)
    # B) Gate-only + Subsampled background (every 2nd, 3rd, 4th sample)
    # C) Full stream training
    print("\n--- The Concept Staleness Experiment (Impact of Training Frequency) ---")
    
    # Let's test a simple margin-based gate (tau = 0.35, which triggers ~15-20% duty)
    tau = 0.30
    gate_mask = ht_margins < tau
    actual_duty = np.mean(gate_mask) * 100.0
    print(f"Testing Margin Gate (tau={tau:.2f}) -> Trips on {actual_duty:.1f}% of samples")
    
    training_regimes = [
        ("Gated-Only (Zero Background)", 0),
        ("Gated + Every 4th Sample (25% Background)", 4),
        ("Gated + Every 3rd Sample (33% Background)", 3),
        ("Gated + Every 2nd Sample (50% Background)", 2),
    ]
    
    for regime_name, bg_step in training_regimes:
        arf_test = ARFClassifier(n_models=10, grace_period=50, lambda_value=6, delta=0.01, seed=42)
        combo_preds = []
        arf_train_count = 0
        
        for i, item in enumerate(samples):
            x, y = item[0], item[1]
            # Predict
            if gate_mask[i]:
                p = arf_test.predict_one(x)
                pred = 1 if p == 1 else 0
            else:
                pred = ht_preds[i]
            combo_preds.append(pred)
            
            # Learn
            should_learn = False
            if gate_mask[i]:
                should_learn = True
            elif bg_step > 0 and (i % bg_step == 0):
                should_learn = True
                
            if should_learn:
                arf_test.learn_one(x, y)
                arf_train_count += 1
                
        combo_acc = np.mean(np.array(combo_preds) == y_trues) * 100.0
        train_duty = arf_train_count / len(samples) * 100.0
        print(f"  Regime: {regime_name:42s} -> Combined Acc: {combo_acc:.2f}%, Training Duty: {train_duty:.1f}%")


def main():
    # Stream 1
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    s1_full = list(loader.iter_stream(warmup_rows=24))
    run_oracle_and_staleness_analysis(s1_full, "Stream 1 (Industrial SCADA)", max_samples=5000)
    
    # Stream 2 (POSOCO)
    s2_full = load_real_posoco_stream("data/india_electricity_real_hourly.csv", warmup_rows=24)
    run_oracle_and_staleness_analysis(s2_full, "Stream 2 (POSOCO Indian Grid)", max_samples=5000)


if __name__ == "__main__":
    main()
