import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import ARFFullWrapper, ARFOptimizedWrapper
from src.controllers.a2_ucb import A2UCBController
from src.analysis.run_day2_benchmarks import A4FixedRouter
from src.controllers.oms_mab import OMSMABController
from src.controllers.cgt_v3 import CGTv3LearnedGateController
from src.models.stronger_baselines import (
    LeveragingBaggingWrapper,
    StreamingRandomPatchesWrapper
)

SEEDS = [42, 101, 7, 2024, 999, 123, 777, 314, 555, 888]

POLICY_NAMES = {
    "always_arf": "Always-ARF (Full, 10 trees)",
    "a3_optimized_arf": "A3: Optimized ARF (5 trees)",
    "srp": "Streaming Random Patches (SRP, 3 trees)",
    "levbag": "Leveraging Bagging (3 trees)",
    "a2_ucb": "A2: Cost-aware UCB",
    "a4_fixed": "A4: Learned Router (Calibrated)",
    "oms_mab": "OMS-MAB (Wilson et al., 2024)",
    "cgt_v3": "CGT-v3 (Learned Meta-Gate)"
}

def build_policy(policy_key, seed):
    if policy_key == "always_arf":
        return ARFFullWrapper(seed=seed)
    elif policy_key == "a3_optimized_arf":
        return ARFOptimizedWrapper(n_models=5, seed=seed)
    elif policy_key == "srp":
        return StreamingRandomPatchesWrapper(n_models=3, grace_period=100, seed=seed)
    elif policy_key == "levbag":
        return LeveragingBaggingWrapper(n_models=3, grace_period=100, seed=seed)
    elif policy_key == "a2_ucb":
        return A2UCBController(c=0.5, discount=0.98, seed=seed)
    elif policy_key == "a4_fixed":
        return A4FixedRouter(batch_size=50, lambda_c=0.02, epsilon=0.05, seed=seed)
    elif policy_key == "oms_mab":
        return OMSMABController(gamma=0.98, c=0.5, seed=seed)
    elif policy_key == "cgt_v3":
        return CGTv3LearnedGateController(
            tau_gate=0.20,
            max_rho=0.10,
            h_cooldown=50,
            buffer_b=200,
            use_pruned_arf=True,
            seed=seed
        )
    raise ValueError(f"Unknown policy {policy_key}")

def evaluate_single_seed_run(policy_key, seed, csv_path, val_len):
    """Worker task executed in a separate process."""
    try:
        loader = ElectricityStreamLoader(csv_path)
        full_stream = list(loader.iter_stream(warmup_rows=24))
        test_stream = full_stream[val_len:]
        n_instances = len(test_stream)
        
        display_name = POLICY_NAMES[policy_key]
        model = build_policy(policy_key, seed)
        
        y_trues = np.zeros(n_instances, dtype=int)
        y_preds = np.zeros(n_instances, dtype=int)
        gate_states = np.zeros(n_instances, dtype=int)
        cpu_times_ns = np.zeros(n_instances, dtype=int)
        
        correct_count = 0
        total_cpu_ns = 0
        
        for idx, (x_t, y_t, ts) in enumerate(test_stream):
            y_trues[idx] = y_t
            
            t_start = time.perf_counter_ns()
            y_pred = model.predict_one(x_t)
            try:
                model.learn_one(x_t, y_t, y_pred)
            except TypeError:
                model.learn_one(x_t, y_t)
                
            t_elapsed = time.perf_counter_ns() - t_start
            cpu_times_ns[idx] = t_elapsed
            y_preds[idx] = y_pred
            
            # Determine gate / ARF activation state
            if hasattr(model, 'gate_open'):
                cur_gate = 1 if model.gate_open else 0
            elif hasattr(model, 'active_arm'):
                cur_gate = 1 if model.active_arm == 'ARF' else 0
            elif hasattr(model, 'selected_arm'):
                cur_gate = 1 if model.selected_arm == 'ARF' else 0
            elif policy_key in ("always_arf", "a3_optimized_arf", "srp", "levbag"):
                cur_gate = 1
            else:
                cur_gate = 0
                
            gate_states[idx] = cur_gate
            total_cpu_ns += t_elapsed
            
            if y_pred == y_t:
                correct_count += 1
                
        acc = (correct_count / n_instances) * 100.0
        mean_cpu_us = (total_cpu_ns / n_instances) / 1000.0
        rho = float(np.mean(gate_states))
        
        # Save raw seed predictions
        out_dir = "results/raw/seeds"
        os.makedirs(out_dir, exist_ok=True)
        raw_file = os.path.join(out_dir, f"{policy_key}_seed_{seed}.csv")
        df_raw = pd.DataFrame({
            'instance_idx': np.arange(1, n_instances + 1),
            'y_true': y_trues,
            'y_pred': y_preds,
            'gate_open': gate_states,
            'cpu_time_ns': cpu_times_ns
        })
        df_raw.to_csv(raw_file, index=False)
        
        return {
            "policy_key": policy_key,
            "policy_name": display_name,
            "seed": seed,
            "accuracy_pct": acc,
            "mean_cpu_us": mean_cpu_us,
            "active_rho": rho,
            "raw_file": raw_file,
            "error": None
        }
    except Exception as e:
        import traceback
        return {
            "policy_key": policy_key,
            "policy_name": POLICY_NAMES.get(policy_key, policy_key),
            "seed": seed,
            "accuracy_pct": 0.0,
            "mean_cpu_us": 0.0,
            "active_rho": 0.0,
            "raw_file": "",
            "error": f"{str(e)}\n{traceback.format_exc()}"
        }

def run_multiseed_benchmark(max_workers=6):
    print("=================================================================", flush=True)
    print("DAY 8 TASK 2: 10-SEED STOCHASTIC BENCHMARK ON FROZEN TEST STREAM", flush=True)
    print("=================================================================", flush=True)
    
    with open("configs/frozen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    csv_path = cfg["data"]["csv_path"]
    val_len = cfg["data"]["val_instances"]
    
    tasks = []
    for p_key in POLICY_NAMES.keys():
        for s in SEEDS:
            tasks.append((p_key, s))
            
    print(f"Total benchmark runs: {len(tasks)} ({len(POLICY_NAMES)} policies x {len(SEEDS)} seeds)", flush=True)
    print(f"Executing with {max_workers} worker processes in parallel...\n", flush=True)
    
    results = []
    t0 = time.perf_counter()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single_seed_run, p_key, s, csv_path, val_len): (p_key, s)
            for (p_key, s) in tasks
        }
        
        completed = 0
        for fut in as_completed(futures):
            p_key, s = futures[fut]
            res = fut.result()
            if res["error"]:
                print(f"ERROR in {p_key} (seed={s}): {res['error']}", flush=True)
            else:
                results.append(res)
                completed += 1
                print(f"[{completed:02d}/{len(tasks):02d}] Completed {res['policy_name']} (seed={res['seed']}) -> Acc: {res['accuracy_pct']:.2f}%, rho: {res['active_rho']:.4f}", flush=True)
            
    t_total = time.perf_counter() - t0
    print(f"\nAll {len(results)} runs completed in {t_total:.1f}s ({t_total/60:.2f} min).\n", flush=True)
    
    # Save individual run results
    df_runs = pd.DataFrame(results)
    os.makedirs("results/tables", exist_ok=True)
    df_runs.to_csv("results/tables/day8_all_seed_runs.csv", index=False)
    
    # Aggregate statistics
    summary_rows = []
    for p_key, p_name in POLICY_NAMES.items():
        sub = df_runs[df_runs["policy_key"] == p_key]
        if len(sub) == 0:
            continue
        accs = sub["accuracy_pct"].values
        rhos = sub["active_rho"].values
        cpus = sub["mean_cpu_us"].values
        
        n = len(accs)
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1)) if n > 1 else 0.0
        
        # Student's t critical value for 95% CI (df = n - 1)
        t_crit = stats.t.ppf(0.975, df=n-1) if n > 1 else 1.96
        se = std_acc / np.sqrt(n)
        ci_low = mean_acc - t_crit * se
        ci_high = mean_acc + t_crit * se
        
        summary_rows.append({
            "policy_key": p_key,
            "policy_name": p_name,
            "seeds_tested": n,
            "mean_acc_pct": round(mean_acc, 2),
            "std_acc_pct": round(std_acc, 2),
            "ci_95_low": round(ci_low, 2),
            "ci_95_high": round(ci_high, 2),
            "ci_95_str": f"[{ci_low:.2f}, {ci_high:.2f}]",
            "min_acc_pct": round(float(np.min(accs)), 2),
            "max_acc_pct": round(float(np.max(accs)), 2),
            "mean_rho": round(float(np.mean(rhos)), 4),
            "mean_cpu_us": round(float(np.mean(cpus)), 2)
        })
        
    df_summary = pd.DataFrame(summary_rows)
    summary_file = "results/tables/day8_multiseed_stability.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"Summary table saved to: {summary_file}")
    print("\n" + df_summary.to_string(index=False))

if __name__ == "__main__":
    run_multiseed_benchmark(max_workers=6)
