import os
import time
import yaml
import hashlib
import numpy as np
import pandas as pd

from src.data.stream_loader import ElectricityStreamLoader
from src.controllers.cgt_v2 import CGTv2Controller
from src.controllers.cgt_v3 import CGTv3LearnedGateController
from src.analysis.run_day2_benchmarks import A4FixedRouter
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper

def evaluate_val_stream(model, val_stream):
    correct = 0
    t_start = time.perf_counter_ns()
    
    for x_t, y_t, ts in val_stream:
        y_pred = model.predict_one(x_t)
        try:
            model.learn_one(x_t, y_t, y_pred)
        except TypeError:
            model.learn_one(x_t, y_t)
        if y_pred == y_t:
            correct += 1
            
    t_total_ns = time.perf_counter_ns() - t_start
    acc = (correct / len(val_stream)) * 100.0
    mean_us = (t_total_ns / len(val_stream)) / 1000.0
    
    return {
        'model_name': getattr(model, 'name', model.__class__.__name__),
        'accuracy_pct': round(acc, 2),
        'mean_cpu_us': round(mean_us, 2),
        'gate_rho': round(getattr(model, 'gate_open_steps', 0) / len(val_stream), 4) if hasattr(model, 'gate_open_steps') else 1.0,
        'alarms': getattr(model, 'total_alarms', 0)
    }

def run_day6_validation_selection():
    print("=================================================================", flush=True)
    print("DAY 6: VALIDATION COMPARISON (CGT-v2 vs CGT-v3 vs A4-Fixed)", flush=True)
    print("=================================================================\n", flush=True)

    with open("configs/frozen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    loader = ElectricityStreamLoader(cfg["data"]["csv_path"])
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_len = cfg["data"]["val_instances"]
    val_stream = full_stream[:val_len]
    seed = cfg["project"]["seed"]

    print(f"Validation Stream: {len(val_stream)} instances (strictly 30% split, mid 2015 – mid 2016).\n", flush=True)

    # 1. Evaluate CGT-v2 baseline on validation
    print("Evaluating CGT-v2 (Adaptive Dual-Gate, margin=0.25, max_rho=0.15, cd=100)...", flush=True)
    cgt_v2 = CGTv2Controller(theta_margin=0.25, max_rho=0.15, h_cooldown=100, seed=seed)
    res_v2 = evaluate_val_stream(cgt_v2, val_stream)
    print(f"  -> CGT-v2: Acc={res_v2['accuracy_pct']}%, rho={res_v2['gate_rho']}, CPU={res_v2['mean_cpu_us']} us/inst", flush=True)

    # 2. Evaluate A4-Fixed router on validation
    print("\nEvaluating A4: Learned Router (Calibrated, lambda_c=0.02, eps=0.05)...", flush=True)
    a4_fixed = A4FixedRouter(batch_size=50, lambda_c=0.02, epsilon=0.05, seed=seed)
    res_a4 = evaluate_val_stream(a4_fixed, val_stream)
    print(f"  -> A4-Fixed: Acc={res_a4['accuracy_pct']}%, rho={res_a4['gate_rho']}, CPU={res_a4['mean_cpu_us']} us/inst", flush=True)

    # 3. Sweep CGT-v3 configurations on validation
    print("\nEvaluating CGT-v3 (Learned Meta-Gate) Sweep on Validation...", flush=True)
    v3_results = []
    tau_candidates = [0.15, 0.20, 0.25, 0.30]
    rho_candidates = [0.10, 0.15]
    cd_candidates = [50, 100]

    for tau in tau_candidates:
        for max_rho in rho_candidates:
            for cd in cd_candidates:
                v3 = CGTv3LearnedGateController(
                    tau_gate=tau,
                    max_rho=max_rho,
                    h_cooldown=cd,
                    buffer_b=200,
                    seed=seed
                )
                res = evaluate_val_stream(v3, val_stream)
                res['tau_gate'] = tau
                res['max_rho'] = max_rho
                res['h_cooldown'] = cd
                v3_results.append(res)
                print(f"  Config [tau={tau}, max_rho={max_rho}, cd={cd}] -> Acc={res['accuracy_pct']}%, rho={res['gate_rho']}, CPU={res['mean_cpu_us']} us/inst", flush=True)

    df_v3 = pd.DataFrame(v3_results)
    os.makedirs("results/tables", exist_ok=True)
    df_v3.to_csv("results/tables/day6_cgt_v3_validation_sweep.csv", index=False)

    best_v3 = df_v3.sort_values(by="accuracy_pct", ascending=False).iloc[0]
    print("\n--------------------------------------------------------------", flush=True)
    print("BEST CGT-v3 CONFIGURATION ON VALIDATION:", flush=True)
    print(f"  Tau Gate: {best_v3['tau_gate']}", flush=True)
    print(f"  Max Rho Budget: {best_v3['max_rho']}", flush=True)
    print(f"  Cooldown Steps: {best_v3['h_cooldown']}", flush=True)
    print(f"  Validation Acc: {best_v3['accuracy_pct']}% (Gate rho={best_v3['gate_rho']})", flush=True)
    print(f"  Validation CPU: {best_v3['mean_cpu_us']} us/inst", flush=True)
    print("--------------------------------------------------------------\n", flush=True)

    # 4. Compare v3 vs v2 vs A4-Fixed
    comparison_rows = [
        {
            "Candidate": "CGT-v2 (Adaptive Dual-Gate)",
            "Validation Acc (%)": res_v2['accuracy_pct'],
            "Active Fraction (rho)": res_v2['gate_rho'],
            "CPU Latency (us/inst)": res_v2['mean_cpu_us'],
            "CPU Saving vs ARF (%)": round((1.0 - (res_v2['mean_cpu_us'] / 3704.66)) * 100, 2),
            "Gate Nature": "Analytical (HT Probability Margin + ADWIN)"
        },
        {
            "Candidate": "CGT-v3 (Learned Meta-Gate)",
            "Validation Acc (%)": best_v3['accuracy_pct'],
            "Active Fraction (rho)": best_v3['gate_rho'],
            "CPU Latency (us/inst)": best_v3['mean_cpu_us'],
            "CPU Saving vs ARF (%)": round((1.0 - (best_v3['mean_cpu_us'] / 3704.66)) * 100, 2),
            "Gate Nature": "Online Logistic Meta-Classifier on 7 features"
        },
        {
            "Candidate": "A4-Fixed (Calibrated Router)",
            "Validation Acc (%)": res_a4['accuracy_pct'],
            "Active Fraction (rho)": res_a4['gate_rho'],
            "CPU Latency (us/inst)": res_a4['mean_cpu_us'],
            "CPU Saving vs ARF (%)": round((1.0 - (res_a4['mean_cpu_us'] / 3704.66)) * 100, 2),
            "Gate Nature": "Cost-aware Contextual Bandit with Exploration"
        }
    ]
    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv("results/tables/day6_gate_selection_comparison.csv", index=False)
    print("Validation Gate Selection Comparison Table:")
    print(df_comp.to_string(index=False), flush=True)

    # 5. Selection Decision: Pick at most ONE final candidate
    # Compare CGT-v3 vs CGT-v2
    # If v3 beats v2 meaningfully on accuracy at similar rho, pick v3.
    # Otherwise, keep v2 or document the difference.
    v3_gain = best_v3['accuracy_pct'] - res_v2['accuracy_pct']
    print(f"\nCGT-v3 vs CGT-v2 Accuracy Delta: {v3_gain:+.2f}%", flush=True)
    
    final_selected = "CGT-v3" if v3_gain >= 0.10 else "CGT-v2"
    print(f"Selected Final Candidate Architecture: {final_selected}\n", flush=True)

    # Freeze configs/frozen_v3.yaml
    frozen_v3_config = {
        "project": {
            "title": "Meta-Controller for Resource-Efficient Data Stream Mining (Config v3)",
            "version": "3.0.0",
            "seed": seed,
            "selected_final_controller": final_selected
        },
        "hypothesis": {
            "delta": 1.0,
            "kappa": 30.0
        },
        "data": {
            "csv_path": cfg["data"]["csv_path"],
            "val_instances": cfg["data"]["val_instances"],
            "test_instances": cfg["data"]["test_instances"]
        },
        "cgt_v3_frozen": {
            "tau_gate": float(best_v3['tau_gate']),
            "max_rho": float(best_v3['max_rho']),
            "h_cooldown": int(best_v3['h_cooldown']),
            "buffer_b": 200,
            "recent_w": 50,
            "use_pruned_arf": True
        },
        "cgt_v2_reference": {
            "theta_margin": 0.25,
            "max_rho": 0.15,
            "h_cooldown": 100,
            "buffer_b": 200,
            "use_pruned_arf": True
        },
        "validation_selection_summary": {
            "v2_accuracy_pct": float(res_v2['accuracy_pct']),
            "v2_gate_rho": float(res_v2['gate_rho']),
            "v3_accuracy_pct": float(best_v3['accuracy_pct']),
            "v3_gate_rho": float(best_v3['gate_rho']),
            "selection_rationale": "Empirical validation comparison with budget cap <= 0.15"
        }
    }

    v3_path = "configs/frozen_v3.yaml"
    with open(v3_path, "w", encoding="utf-8") as f:
        yaml.dump(frozen_v3_config, f, default_flow_style=False, sort_keys=False)

    with open(v3_path, "rb") as f:
        v3_hash = hashlib.sha256(f.read()).hexdigest()

    v3_hash_path = "configs/frozen_v3_hash.txt"
    with open(v3_hash_path, "w", encoding="utf-8") as f:
        f.write(f"SHA-256: {v3_hash}\n")

    print(f"Successfully locked {v3_path} with SHA-256: {v3_hash}", flush=True)
    return best_v3, res_v2, v3_hash

if __name__ == "__main__":
    run_day6_validation_selection()

