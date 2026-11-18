import os
import time
import yaml
import hashlib
import numpy as np
import pandas as pd

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper, ARFOptimizedWrapper
from src.controllers.cgt import ConditionalGatedTrainingController
from src.controllers.oms_mab import OMSMABController
from src.controllers.cgt_v2 import CGTv2Controller

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
        'name': getattr(model, 'name', model.__class__.__name__),
        'accuracy_pct': round(acc, 2),
        'mean_cpu_us': round(mean_us, 2),
        'gate_rho': round(getattr(model, 'gate_open_steps', 0) / len(val_stream), 4) if hasattr(model, 'gate_open_steps') else 1.0,
        'alarms': getattr(model, 'total_alarms', 0)
    }

def run_tuning_pipeline():
    print("=================================================================", flush=True)
    print("DAY 5: CONTROLLER OPTIMIZATION ON 30% VALIDATION SPLIT ONLY", flush=True)
    print("=================================================================\n", flush=True)

    with open("configs/frozen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    loader = ElectricityStreamLoader(cfg["data"]["csv_path"])
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_len = cfg["data"]["val_instances"]
    val_stream = full_stream[:val_len]
    seed = cfg["project"]["seed"]

    print(f"Loaded validation stream: {len(val_stream)} instances (strictly disjoint from test stream).\n", flush=True)

    # 1. Evaluate Baselines on Validation Split
    print("Evaluating Baselines on Validation Split...", flush=True)
    ht_val = evaluate_val_stream(HoeffdingTreeWrapper(), val_stream)
    print(f"  -> Always-HT: Acc={ht_val['accuracy_pct']}%, CPU={ht_val['mean_cpu_us']} us/inst", flush=True)

    arf_val = evaluate_val_stream(ARFFullWrapper(seed=seed), val_stream)
    print(f"  -> Always-ARF: Acc={arf_val['accuracy_pct']}%, CPU={arf_val['mean_cpu_us']} us/inst", flush=True)

    cgt_v1_val = evaluate_val_stream(
        ConditionalGatedTrainingController(
            stability_k=cfg["cgt_frozen"]["stability_k"],
            buffer_b=cfg["cgt_frozen"]["buffer_b"],
            warm_start_strategy="S1",
            adwin_delta=cfg["cgt_frozen"]["adwin_delta"],
            adwin_clock=cfg["cgt_frozen"]["adwin_clock"],
            seed=seed
        ),
        val_stream
    )
    print(f"  -> CGT-v1: Acc={cgt_v1_val['accuracy_pct']}%, rho={cgt_v1_val['gate_rho']}, CPU={cgt_v1_val['mean_cpu_us']} us/inst", flush=True)

    oms_val = evaluate_val_stream(OMSMABController(gamma=0.98, c=0.5, seed=seed), val_stream)
    print(f"  -> OMS-MAB: Acc={oms_val['accuracy_pct']}%, rho={oms_val['gate_rho']}, CPU={oms_val['mean_cpu_us']} us/inst", flush=True)

    # 2. Grid Sweep for CGT-v2 on Validation Split
    print("\nRunning Hyperparameter Sweep for CGT-v2 on Validation Split...", flush=True)
    sweep_results = []
    
    margins = [0.10, 0.15, 0.20, 0.25]
    budgets = [0.15, 0.20, 0.25]
    cooldowns = [50, 100]

    for m in margins:
        for b in budgets:
            for cd in cooldowns:
                ctrl = CGTv2Controller(
                    theta_margin=m,
                    max_rho=b,
                    h_cooldown=cd,
                    buffer_b=200,
                    adwin_delta=0.005,
                    adwin_clock=32,
                    use_pruned_arf=True,
                    seed=seed
                )
                res = evaluate_val_stream(ctrl, val_stream)
                res['theta_margin'] = m
                res['max_rho'] = b
                res['h_cooldown'] = cd
                sweep_results.append(res)
                print(f"  Config [margin={m}, max_rho={b}, cd={cd}] -> Acc={res['accuracy_pct']}%, rho={res['gate_rho']}, CPU={res['mean_cpu_us']} us/inst", flush=True)

    df_sweep = pd.DataFrame(sweep_results)
    os.makedirs("results/tables", exist_ok=True)
    df_sweep.to_csv("results/tables/day5_cgt_v2_validation_sweep.csv", index=False)
    print(f"\nSaved validation sweep results to results/tables/day5_cgt_v2_validation_sweep.csv", flush=True)

    # 3. Select Best Configuration
    # Filter for candidates that satisfy resource constraint: gate_rho <= 0.20 (at least 80% theoretical CPU savings)
    valid_candidates = df_sweep[df_sweep['gate_rho'] <= 0.20]
    if valid_candidates.empty:
        best_cfg_row = df_sweep.sort_values(by="accuracy_pct", ascending=False).iloc[0]
    else:
        best_cfg_row = valid_candidates.sort_values(by="accuracy_pct", ascending=False).iloc[0]

    print("\n--------------------------------------------------------------", flush=True)
    print("OPTIMAL VALIDATION CONFIGURATION FOR CGT-v2:", flush=True)
    print(f"  Theta Margin: {best_cfg_row['theta_margin']}", flush=True)
    print(f"  Max ARF Budget (max_rho): {best_cfg_row['max_rho']}", flush=True)
    print(f"  Cooldown Steps: {best_cfg_row['h_cooldown']}", flush=True)
    print(f"  Validation Accuracy: {best_cfg_row['accuracy_pct']}% (vs CGT-v1 {cgt_v1_val['accuracy_pct']}%)", flush=True)
    print(f"  Validation Gate Fraction: {best_cfg_row['gate_rho']}", flush=True)
    print(f"  Validation CPU Latency: {best_cfg_row['mean_cpu_us']} us/inst", flush=True)
    print("--------------------------------------------------------------\n", flush=True)

    # 4. Freeze Config v2 with SHA-256
    frozen_v2_config = {
        "project": {
            "title": "Meta-Controller for Resource-Efficient Data Stream Mining (Config v2)",
            "version": "2.0.0",
            "seed": seed
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
        "cgt_v2_frozen": {
            "theta_margin": float(best_cfg_row['theta_margin']),
            "max_rho": float(best_cfg_row['max_rho']),
            "h_cooldown": int(best_cfg_row['h_cooldown']),
            "buffer_b": 200,
            "adwin_delta": 0.005,
            "adwin_clock": 32,
            "use_pruned_arf": True
        },
        "oms_mab_frozen": {
            "gamma": 0.98,
            "c": 0.5
        },
        "validation_metrics": {
            "accuracy_pct": float(best_cfg_row['accuracy_pct']),
            "gate_rho": float(best_cfg_row['gate_rho']),
            "mean_cpu_us": float(best_cfg_row['mean_cpu_us'])
        }
    }

    v2_path = "configs/frozen_v2.yaml"
    with open(v2_path, "w", encoding="utf-8") as f:
        yaml.dump(frozen_v2_config, f, default_flow_style=False, sort_keys=False)

    with open(v2_path, "rb") as f:
        v2_hash = hashlib.sha256(f.read()).hexdigest()

    v2_hash_path = "configs/frozen_v2_hash.txt"
    with open(v2_hash_path, "w", encoding="utf-8") as f:
        f.write(f"SHA-256: {v2_hash}\n")

    print(f"Successfully created and locked {v2_path}", flush=True)
    print(f"Config v2 SHA-256 Hash: {v2_hash}", flush=True)
    print(f"Hash saved to {v2_hash_path}\n", flush=True)

    # 5. Build and Save Validation Comparison Table
    df_val_comp = pd.DataFrame([
        {
            "Model / Controller": "Always-HT",
            "Validation Acc (%)": ht_val['accuracy_pct'],
            "Gate Fraction (rho)": 0.0000,
            "Mean CPU (us/inst)": ht_val['mean_cpu_us'],
            "Role": "Lightweight Single-Tree Baseline"
        },
        {
            "Model / Controller": "CGT-v1 (Proposed, S1)",
            "Validation Acc (%)": cgt_v1_val['accuracy_pct'],
            "Gate Fraction (rho)": cgt_v1_val['gate_rho'],
            "Mean CPU (us/inst)": cgt_v1_val['mean_cpu_us'],
            "Role": "Pure ADWIN Drift-Gated Switch"
        },
        {
            "Model / Controller": "OMS-MAB (Wilson et al., 2024)",
            "Validation Acc (%)": oms_val['accuracy_pct'],
            "Gate Fraction (rho)": oms_val['gate_rho'],
            "Mean CPU (us/inst)": oms_val['mean_cpu_us'],
            "Role": "Multi-Armed Bandit Online Model Selection"
        },
        {
            "Model / Controller": "CGT-v2 (Adaptive Dual-Gate)",
            "Validation Acc (%)": best_cfg_row['accuracy_pct'],
            "Gate Fraction (rho)": best_cfg_row['gate_rho'],
            "Mean CPU (us/inst)": best_cfg_row['mean_cpu_us'],
            "Role": "Dual Macro ADWIN + Micro Margin Gating"
        },
        {
            "Model / Controller": "Always-ARF (10 trees)",
            "Validation Acc (%)": arf_val['accuracy_pct'],
            "Gate Fraction (rho)": 1.0000,
            "Mean CPU (us/inst)": arf_val['mean_cpu_us'],
            "Role": "Heavy Continuous Ensemble Reference"
        }
    ])
    df_val_comp.to_csv("results/tables/day5_validation_comparison.csv", index=False)
    print("Validation Comparison Table:")
    print(df_val_comp.to_string(index=False), flush=True)

    return v2_hash

if __name__ == "__main__":
    run_tuning_pipeline()

