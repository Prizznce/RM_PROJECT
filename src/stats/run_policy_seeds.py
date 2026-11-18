import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
import numpy as np
import pandas as pd
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

POLICY_BUILDERS = {
    "always_arf": ("Always-ARF (Full, 10 trees)", lambda s: ARFFullWrapper(seed=s)),
    "a3_optimized_arf": ("A3: Optimized ARF (5 trees)", lambda s: ARFOptimizedWrapper(n_models=5, seed=s)),
    "srp": ("Streaming Random Patches (SRP, 3 trees)", lambda s: StreamingRandomPatchesWrapper(n_models=3, grace_period=100, seed=s)),
    "levbag": ("Leveraging Bagging (3 trees)", lambda s: LeveragingBaggingWrapper(n_models=3, grace_period=100, seed=s)),
    "a2_ucb": ("A2: Cost-aware UCB", lambda s: A2UCBController(c=0.5, discount=0.98, seed=s)),
    "a4_fixed": ("A4: Learned Router (Calibrated)", lambda s: A4FixedRouter(batch_size=50, lambda_c=0.02, epsilon=0.05, seed=s)),
    "oms_mab": ("OMS-MAB (Wilson et al., 2024)", lambda s: OMSMABController(gamma=0.98, c=0.5, seed=s)),
    "cgt_v3": ("CGT-v3 (Learned Meta-Gate)", lambda s: CGTv3LearnedGateController(
        tau_gate=0.20,
        max_rho=0.10,
        h_cooldown=50,
        buffer_b=200,
        use_pruned_arf=True,
        seed=s
    ))
}

def run_policy_for_all_seeds(policy_key):
    if policy_key not in POLICY_BUILDERS:
        print(f"Error: Unknown policy key '{policy_key}'", flush=True)
        return

    display_name, factory = POLICY_BUILDERS[policy_key]
    print(f"==================================================", flush=True)
    print(f"Executing 10 Seeds for: [{policy_key}] {display_name}", flush=True)
    print(f"==================================================", flush=True)

    with open("configs/frozen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    csv_path = cfg["data"]["csv_path"]
    val_len = cfg["data"]["val_instances"]

    print("Loading test stream...", flush=True)
    loader = ElectricityStreamLoader(csv_path)
    full_stream = list(loader.iter_stream(warmup_rows=24))
    test_stream = full_stream[val_len:]
    n_instances = len(test_stream)
    print(f"Loaded {n_instances} test instances.\n", flush=True)

    out_dir = "results/raw/seeds"
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for seed_idx, seed in enumerate(SEEDS, 1):
        raw_file = os.path.join(out_dir, f"{policy_key}_seed_{seed}.csv")
        
        # Check if already computed
        if os.path.exists(raw_file):
            print(f"[{seed_idx}/{len(SEEDS)}] Seed {seed} already computed. Loading...", flush=True)
            df_existing = pd.read_csv(raw_file)
            correct = (df_existing['y_true'] == df_existing['y_pred']).sum()
            acc = (correct / len(df_existing)) * 100.0
            rho = float(df_existing['gate_open'].mean())
            mean_cpu = float(df_existing['cpu_time_ns'].mean() / 1000.0) if 'cpu_time_ns' in df_existing else 0.0
            records.append({
                "policy_key": policy_key,
                "policy_name": display_name,
                "seed": seed,
                "accuracy_pct": acc,
                "mean_cpu_us": mean_cpu,
                "active_rho": rho,
                "raw_file": raw_file
            })
            print(f"  -> Acc: {acc:.2f}%, rho: {rho:.4f}", flush=True)
            continue

        print(f"[{seed_idx}/{len(SEEDS)}] Running seed {seed}...", flush=True)
        t0 = time.perf_counter()
        model = factory(seed)

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

        t_wall = time.perf_counter() - t0
        acc = (correct_count / n_instances) * 100.0
        mean_cpu_us = (total_cpu_ns / n_instances) / 1000.0
        rho = float(np.mean(gate_states))

        df_raw = pd.DataFrame({
            'instance_idx': np.arange(1, n_instances + 1),
            'y_true': y_trues,
            'y_pred': y_preds,
            'gate_open': gate_states,
            'cpu_time_ns': cpu_times_ns
        })
        df_raw.to_csv(raw_file, index=False)

        records.append({
            "policy_key": policy_key,
            "policy_name": display_name,
            "seed": seed,
            "accuracy_pct": acc,
            "mean_cpu_us": mean_cpu_us,
            "active_rho": rho,
            "raw_file": raw_file
        })
        print(f"  -> Acc: {acc:.2f}%, rho: {rho:.4f}, CPU: {mean_cpu_us:.1f} us/inst (Wall: {t_wall:.1f}s)", flush=True)

    df_policy = pd.DataFrame(records)
    partial_csv = f"results/tables/seeds_{policy_key}.csv"
    df_policy.to_csv(partial_csv, index=False)
    print(f"\nSaved partial results to {partial_csv}", flush=True)
    print(f"Mean Acc: {df_policy['accuracy_pct'].mean():.2f}% +/- {df_policy['accuracy_pct'].std():.2f}%\n", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.stats.run_policy_seeds <policy_key_or_all>")
        sys.exit(1)
        
    target = sys.argv[1]
    if target == "all":
        for p in POLICY_BUILDERS.keys():
            run_policy_for_all_seeds(p)
    else:
        run_policy_for_all_seeds(target)

