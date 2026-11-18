import os
import time
import yaml
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper, ARFOptimizedWrapper
from src.controllers.cgt import ConditionalGatedTrainingController
from src.controllers.oracle_switch import OracleGatedSwitch
from src.controllers.a4_router import A4LearnedRouter
from src.analysis.run_day2_benchmarks import A4FixedRouter
from src.models.stronger_baselines import (
    NaivePersistenceHourly,
    NaivePersistenceDiurnal,
    ZeroSkillMajorityClass,
    StreamingSGDClassifier,
    AdaptiveHoeffdingTreeWrapper,
    StreamingKNNWrapper,
    LeveragingBaggingWrapper,
    StreamingRandomPatchesWrapper
)

def evaluate_single_model(model, stream):
    process = psutil.Process()
    initial_mem_mb = process.memory_info().rss / (1024 * 1024)
    peak_mem_mb = initial_mem_mb
    
    total_instances = 0
    correct_count = 0
    total_cpu_ns = 0
    
    is_oracle = isinstance(model, OracleGatedSwitch)
    
    for x_t, y_t, ts in stream:
        total_instances += 1
        
        t_start = time.perf_counter_ns()
        
        if is_oracle:
            y_pred, _ = model.evaluate_and_learn(x_t, y_t)
        else:
            y_pred = model.predict_one(x_t)
            try:
                model.learn_one(x_t, y_t, y_pred)
            except TypeError:
                model.learn_one(x_t, y_t)
                
        t_elapsed = time.perf_counter_ns() - t_start
        total_cpu_ns += t_elapsed
        
        if y_pred == y_t:
            correct_count += 1
            
        if total_instances % 1000 == 0:
            current_mem = process.memory_info().rss / (1024 * 1024)
            if current_mem > peak_mem_mb:
                peak_mem_mb = current_mem

    overall_acc = (correct_count / total_instances) * 100.0
    mean_cpu_us = (total_cpu_ns / total_instances) / 1000.0
    
    return {
        'model_name': getattr(model, 'name', model.__class__.__name__),
        'total_instances': total_instances,
        'accuracy_pct': round(overall_acc, 2),
        'mean_cpu_us_per_instance': round(mean_cpu_us, 2),
        'total_cpu_sec': round(total_cpu_ns / 1e9, 3),
        'peak_memory_mb': round(peak_mem_mb, 2)
    }

def run_day3_extended_benchmark():
    print("=================================================================", flush=True)
    print("DAY 3: COMPREHENSIVE BENCHMARKING ACROSS EXPANDED MODEL FAMILIES", flush=True)
    print("=================================================================\n", flush=True)

    with open("configs/frozen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    loader = ElectricityStreamLoader(cfg["data"]["csv_path"])
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_len = cfg["data"]["val_instances"]
    test_stream = full_stream[val_len:]
    seed = cfg["project"]["seed"]

    print(f"Total frozen test stream: {len(test_stream)} instances\n", flush=True)

    models_to_evaluate = [
        # 1. Naive / Autoregressive Baselines
        (ZeroSkillMajorityClass(), "Naive / Autoregressive"),
        (NaivePersistenceHourly(), "Naive / Autoregressive"),
        (NaivePersistenceDiurnal(), "Naive / Autoregressive"),
        
        # 2. Linear & Instance-based
        (StreamingSGDClassifier(lr=0.01), "Online Linear (SGD)"),
        (StreamingKNNWrapper(n_neighbors=5, window_size=100), "Instance-Based (kNN)"),
        
        # 3. Single Decision Trees
        (HoeffdingTreeWrapper(), "Single Streaming Tree"),
        (AdaptiveHoeffdingTreeWrapper(grace_period=200, seed=seed), "Single Streaming Tree"),
        
        # 4. Ensembles
        (LeveragingBaggingWrapper(n_models=3, grace_period=100, seed=seed), "Streaming Ensemble"),
        (StreamingRandomPatchesWrapper(n_models=3, grace_period=100, seed=seed), "Streaming Ensemble"),
        (ARFOptimizedWrapper(n_models=5, seed=seed), "Streaming Ensemble"),
        (ARFFullWrapper(seed=seed), "Streaming Ensemble"),
        
        # 5. Dynamic Meta-Controllers
        (A4LearnedRouter(batch_size=50, lambda_c=0.3, seed=seed), "Meta-Controller"),
        (A4FixedRouter(batch_size=50, lambda_c=0.02, epsilon=0.05, seed=seed), "Meta-Controller"),
        (ConditionalGatedTrainingController(
            stability_k=cfg["cgt_frozen"]["stability_k"],
            buffer_b=cfg["cgt_frozen"]["buffer_b"],
            warm_start_strategy="S1",
            adwin_delta=cfg["cgt_frozen"]["adwin_delta"],
            adwin_clock=cfg["cgt_frozen"]["adwin_clock"],
            seed=seed
        ), "Meta-Controller"),
        (ConditionalGatedTrainingController(
            stability_k=1000,
            buffer_b=300,
            warm_start_strategy="S1",
            adwin_delta=0.02,
            adwin_clock=16,
            use_pruned_arf=True,
            seed=seed
        ), "Meta-Controller"),
        
        # 6. Theoretical Upper Bound
        (OracleGatedSwitch(seed=seed), "Omniscient Ceiling")
    ]

    results = []
    for idx, (model, family) in enumerate(models_to_evaluate, 1):
        print(f"[{idx}/{len(models_to_evaluate)}] Evaluating [{family}] {model.name}...", flush=True)
        t0 = time.perf_counter()
        res = evaluate_single_model(model, test_stream)
        t_elapsed = time.perf_counter() - t0
        res['model_family'] = family
        results.append(res)
        print(f"    -> Acc: {res['accuracy_pct']}%, CPU: {res['mean_cpu_us_per_instance']} us/inst, Wall Time: {t_elapsed:.1f}s", flush=True)

    df_res = pd.DataFrame(results)

    # Reference Always-ARF latency for savings calculation
    arf_row = df_res[df_res['model_name'] == 'Always-ARF'].iloc[0]
    arf_cpu = arf_row['mean_cpu_us_per_instance']

    df_res['cpu_saving_pct'] = df_res['mean_cpu_us_per_instance'].apply(
        lambda c: round((1.0 - (c / arf_cpu)) * 100.0, 2)
    )

    # Reorder columns
    cols = ['model_family', 'model_name', 'accuracy_pct', 'mean_cpu_us_per_instance', 'cpu_saving_pct', 'total_cpu_sec', 'peak_memory_mb']
    df_res = df_res[cols]

    out_csv = "results/tables/day3_comprehensive_baselines.csv"
    os.makedirs("results/tables", exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved comprehensive Day 3 benchmark table to {out_csv}", flush=True)
    print("\n" + df_res.to_string(index=False), flush=True)

    # Generate Publication-Ready Pareto Chart
    generate_day3_pareto_plot(df_res)

    return df_res

def generate_day3_pareto_plot(df):
    os.makedirs("results/figures", exist_ok=True)
    plt.figure(figsize=(13, 8))

    color_map = {
        "Naive / Autoregressive": "#7f7f7f",
        "Online Linear (SGD)": "#8c564b",
        "Instance-Based (kNN)": "#e377c2",
        "Single Streaming Tree": "#1f77b4",
        "Streaming Ensemble": "#2ca02c",
        "Meta-Controller": "#ff7f0e",
        "Omniscient Ceiling": "#d62728"
    }

    marker_map = {
        "Naive / Autoregressive": "X",
        "Online Linear (SGD)": "s",
        "Instance-Based (kNN)": "p",
        "Single Streaming Tree": "o",
        "Streaming Ensemble": "^",
        "Meta-Controller": "D",
        "Omniscient Ceiling": "*"
    }

    for family, group in df.groupby("model_family"):
        plt.scatter(
            group["mean_cpu_us_per_instance"],
            group["accuracy_pct"],
            c=color_map.get(family, "black"),
            marker=marker_map.get(family, "o"),
            s=130 if family != "Omniscient Ceiling" else 260,
            alpha=0.9,
            edgecolors="black",
            linewidth=1.0,
            label=family
        )
        for _, row in group.iterrows():
            offset_x = 1.08
            offset_y = 0.15
            if "Oracle" in row["model_name"]:
                offset_y = 0.35
            elif "CGT-Plus" in row["model_name"]:
                offset_y = -0.55
            elif "Always-HT" in row["model_name"]:
                offset_y = -0.55
            elif "Majority" in row["model_name"]:
                offset_y = 0.3
            elif "kNN" in row["model_name"]:
                offset_y = -0.55

            plt.annotate(
                row["model_name"],
                (row["mean_cpu_us_per_instance"] * offset_x, row["accuracy_pct"] + offset_y),
                fontsize=8.5,
                fontweight="bold" if ("CGT" in row["model_name"] or "Oracle" in row["model_name"]) else "normal"
            )

    plt.xscale("log")
    plt.xlabel("Mean CPU Latency per Instance (µs, Log Scale)", fontsize=12)
    plt.ylabel("Prequential Sliding Accuracy (%)", fontsize=12)
    plt.title("Empirical Pareto Frontier: Accuracy vs. Computational Cost Across Model Families", fontsize=13, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.ylim(52, 95)

    out_fig = "results/figures/fig_day3_extended_baselines_pareto.png"
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Generated Day 3 Extended Pareto Figure: {out_fig}", flush=True)

if __name__ == "__main__":
    run_day3_extended_benchmark()
