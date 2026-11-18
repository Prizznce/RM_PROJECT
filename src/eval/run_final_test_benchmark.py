import os
import time
import yaml
import psutil
import numpy as np
import pandas as pd

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper, ARFOptimizedWrapper
from src.controllers.a1_threshold import A1ThresholdController
from src.controllers.a2_ucb import A2UCBController
from src.analysis.run_day2_benchmarks import A4FixedRouter
from src.controllers.oms_mab import OMSMABController
from src.controllers.cgt import ConditionalGatedTrainingController
from src.controllers.cgt_v2 import CGTv2Controller
from src.controllers.cgt_v3 import CGTv3LearnedGateController
from src.controllers.oracle_switch import OracleGatedSwitch
from src.models.stronger_baselines import (
    NaivePersistenceHourly,
    ZeroSkillMajorityClass,
    StreamingSGDClassifier,
    AdaptiveHoeffdingTreeWrapper,
    StreamingKNNWrapper,
    LeveragingBaggingWrapper,
    StreamingRandomPatchesWrapper
)

def evaluate_and_log_policy(model, family, test_stream, out_dir="results/raw"):
    os.makedirs(out_dir, exist_ok=True)
    m_name = getattr(model, 'name', model.__class__.__name__)
    safe_name = m_name.replace(":", "").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").lower()
    log_csv = os.path.join(out_dir, f"step_predictions_{safe_name}.csv")
    
    process = psutil.Process()
    mem_start_mb = process.memory_info().rss / (1024 * 1024)
    
    is_oracle = isinstance(model, OracleGatedSwitch)
    
    n_instances = len(test_stream)
    y_trues = np.zeros(n_instances, dtype=int)
    y_preds = np.zeros(n_instances, dtype=int)
    gate_states = np.zeros(n_instances, dtype=int)
    cpu_times_ns = np.zeros(n_instances, dtype=int)
    
    total_cpu_ns = 0
    correct_count = 0
    
    for idx, (x_t, y_t, ts) in enumerate(test_stream):
        y_trues[idx] = y_t
        
        # Controlled Timing Measurement Block (predict + learn)
        # Standardized protocol: No file I/O or psutil calls inside timer
        t_start = time.perf_counter_ns()
        
        if is_oracle:
            y_pred, arm = model.evaluate_and_learn(x_t, y_t)
            current_gate = 1 if arm == "ARF" else 0
        else:
            y_pred = model.predict_one(x_t)
            try:
                model.learn_one(x_t, y_t, y_pred)
            except TypeError:
                model.learn_one(x_t, y_t)
            current_gate = 1 if getattr(model, 'gate_open', False) else (1 if getattr(model, 'active_arm', '') == 'ARF' else 0)
            
        t_elapsed = time.perf_counter_ns() - t_start
        
        cpu_times_ns[idx] = t_elapsed
        y_preds[idx] = y_pred
        gate_states[idx] = current_gate
        total_cpu_ns += t_elapsed
        
        if y_pred == y_t:
            correct_count += 1

    mem_end_mb = process.memory_info().rss / (1024 * 1024)
    peak_mem_mb = max(mem_start_mb, mem_end_mb)
    
    overall_acc = (correct_count / n_instances) * 100.0
    mean_cpu_us = (total_cpu_ns / n_instances) / 1000.0
    active_gate_rho = float(np.mean(gate_states))
    
    # Save raw per-instance logs
    df_raw = pd.DataFrame({
        'instance_idx': np.arange(1, n_instances + 1),
        'y_true': y_trues,
        'y_pred': y_preds,
        'gate_open': gate_states,
        'cpu_time_ns': cpu_times_ns
    })
    df_raw.to_csv(log_csv, index=False)
    
    return {
        'model_family': family,
        'model_name': m_name,
        'accuracy_pct': round(overall_acc, 2),
        'mean_cpu_us_per_instance': round(mean_cpu_us, 2),
        'active_gate_rho': round(active_gate_rho, 4),
        'total_cpu_sec': round(total_cpu_ns / 1e9, 3),
        'peak_memory_mb': round(peak_mem_mb, 2),
        'raw_log_file': log_csv
    }

def run_day6_clean_benchmark():
    print("=================================================================", flush=True)
    print("DAY 6: CLEAN FROZEN TEST STREAM BENCHMARK (CONTROLLED TIMING PROTOCOL)", flush=True)
    print("=================================================================\n", flush=True)

    with open("configs/frozen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    loader = ElectricityStreamLoader(cfg["data"]["csv_path"])
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_len = cfg["data"]["val_instances"]
    test_stream = full_stream[val_len:]
    seed = cfg["project"]["seed"]

    print(f"Test Stream Size: {len(test_stream)} instances (strictly evaluated once under frozen parameters).\n", flush=True)

    all_policies = [
        # 1. Baselines
        (ZeroSkillMajorityClass(), "Naive Baseline"),
        (NaivePersistenceHourly(), "Naive Baseline"),
        (StreamingSGDClassifier(lr=0.01), "Online Linear"),
        (StreamingKNNWrapper(n_neighbors=5, window_size=100), "Instance-Based"),
        
        # 2. Single Streaming Trees
        (HoeffdingTreeWrapper(), "Single Tree"),
        (AdaptiveHoeffdingTreeWrapper(grace_period=200, seed=seed), "Single Tree"),
        
        # 3. Ensembles
        (LeveragingBaggingWrapper(n_models=3, grace_period=100, seed=seed), "Ensemble"),
        (StreamingRandomPatchesWrapper(n_models=3, grace_period=100, seed=seed), "Ensemble"),
        (A1ThresholdController(theta_hi=0.25, theta_lo=0.15, window_size=50, seed=seed), "Switching Policy"),
        (A2UCBController(c=0.5, discount=0.98, seed=seed), "Switching Policy"),
        (ARFOptimizedWrapper(n_models=5, seed=seed), "Ensemble"),
        (ARFFullWrapper(seed=seed), "Ensemble Reference"),
        
        # 4. Meta-Controllers & Bandits
        (A4FixedRouter(batch_size=50, lambda_c=0.02, epsilon=0.05, seed=seed), "Learned Router"),
        (OMSMABController(gamma=0.98, c=0.5, seed=seed), "Bandit Baseline (OMS-MAB)"),
        (ConditionalGatedTrainingController(
            stability_k=cfg["cgt_frozen"]["stability_k"],
            buffer_b=cfg["cgt_frozen"]["buffer_b"],
            warm_start_strategy="S1",
            adwin_delta=cfg["cgt_frozen"]["adwin_delta"],
            adwin_clock=cfg["cgt_frozen"]["adwin_clock"],
            seed=seed
        ), "CGT Baseline (v1)"),
        (CGTv2Controller(
            theta_margin=0.25,
            max_rho=0.15,
            h_cooldown=100,
            buffer_b=200,
            use_pruned_arf=True,
            seed=seed
        ), "CGT Dual-Gate (v2)"),
        (CGTv3LearnedGateController(
            tau_gate=0.20,
            max_rho=0.10,
            h_cooldown=50,
            buffer_b=200,
            use_pruned_arf=True,
            seed=seed
        ), "CGT Learned-Gate (v3)"),
        
        # 5. Theoretical Upper Bound
        (OracleGatedSwitch(seed=seed), "Omniscient Ceiling")
    ]

    results = []
    for idx, (policy, family) in enumerate(all_policies, 1):
        print(f"[{idx}/{len(all_policies)}] Evaluating [{family}] {policy.name}...", flush=True)
        t0 = time.perf_counter()
        res = evaluate_and_log_policy(policy, family, test_stream)
        t_elapsed = time.perf_counter() - t0
        results.append(res)
        print(f"    -> Acc: {res['accuracy_pct']}%, rho: {res['active_gate_rho']}, CPU: {res['mean_cpu_us_per_instance']} us/inst (Wall: {t_elapsed:.1f}s)", flush=True)

    df_res = pd.DataFrame(results)

    # Reference Always-ARF latency
    arf_row = df_res[df_res['model_name'] == 'Always-ARF'].iloc[0]
    arf_cpu = arf_row['mean_cpu_us_per_instance']

    df_res['cpu_saving_pct'] = df_res['mean_cpu_us_per_instance'].apply(
        lambda c: round((1.0 - (c / arf_cpu)) * 100.0, 2)
    )

    cols = ['model_family', 'model_name', 'accuracy_pct', 'active_gate_rho', 'mean_cpu_us_per_instance', 'cpu_saving_pct', 'total_cpu_sec', 'peak_memory_mb', 'raw_log_file']
    df_res = df_res[cols]

    out_csv = "results/tables/day6_clean_test_benchmark.csv"
    os.makedirs("results/tables", exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    
    print(f"\n=================================================================", flush=True)
    print(f"CLEAN FROZEN BENCHMARK COMPLETE. Saved to {out_csv}", flush=True)
    print(f"=================================================================\n", flush=True)
    print(df_res[['model_family', 'model_name', 'accuracy_pct', 'active_gate_rho', 'mean_cpu_us_per_instance', 'cpu_saving_pct']].to_string(index=False), flush=True)

    return df_res

if __name__ == "__main__":
    run_day6_clean_benchmark()

