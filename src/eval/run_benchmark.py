import os
import yaml
import time
import pandas as pd
import numpy as np
from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper, ARFOptimizedWrapper
from src.controllers.a1_threshold import A1ThresholdController
from src.controllers.a2_ucb import A2UCBController
from src.controllers.a4_router import A4LearnedRouter
from src.controllers.cgt import ConditionalGatedTrainingController
from src.controllers.cgt_ablation import CGTFixedScheduleAblation
from src.eval.prequential import run_prequential_evaluation

def run_benchmark():
    # 1. Load frozen configuration
    frozen_config_path = "configs/frozen.yaml"
    if not os.path.exists(frozen_config_path):
        raise FileNotFoundError("configs/frozen.yaml must exist before running benchmark!")
    with open(frozen_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Load stream and obtain the frozen test split (70%)
    loader = ElectricityStreamLoader(cfg["data"]["csv_path"])
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_len = cfg["data"]["val_instances"]
    test_stream = full_stream[val_len:]
    
    print("=================================================================")
    print("RUNNING OPTIMIZED BENCHMARK ON FROZEN TEST STREAM (70% SPLIT)")
    print(f"Total Test Stream Length: {len(test_stream)} instances")
    print("=================================================================\n")

    # Models to benchmark
    cgt_cfg = cfg["cgt_frozen"]
    a1_cfg = cfg["a1_frozen"]
    a2_cfg = cfg["a2_frozen"]
    a4_cfg = cfg["a4_frozen"]
    
    models = [
        HoeffdingTreeWrapper(),
        ARFFullWrapper(seed=cfg["project"]["seed"]),
        A1ThresholdController(
            theta_hi=a1_cfg["theta_hi"],
            theta_lo=a1_cfg["theta_lo"],
            window_size=a1_cfg["window_size"],
            seed=cfg["project"]["seed"]
        ),
        A2UCBController(
            c=a2_cfg["c"],
            lambda_acc=a2_cfg["lambda_acc"],
            lambda_v=a2_cfg["lambda_v"],
            lambda_d=a2_cfg["lambda_d"],
            lambda_c=a2_cfg["lambda_c"],
            discount=a2_cfg["discount"],
            seed=cfg["project"]["seed"]
        ),
        ARFOptimizedWrapper(seed=cfg["project"]["seed"]),
        A4LearnedRouter(
            batch_size=a4_cfg["batch_size"],
            lambda_c=a4_cfg["lambda_c"],
            seed=cfg["project"]["seed"]
        ),
        ConditionalGatedTrainingController(
            stability_k=cgt_cfg["stability_k"],
            buffer_b=cgt_cfg["buffer_b"],
            warm_start_strategy="S1",
            adwin_delta=cgt_cfg["adwin_delta"],
            adwin_clock=cgt_cfg["adwin_clock"],
            seed=cfg["project"]["seed"]
        ),
        ConditionalGatedTrainingController(
            stability_k=1000,
            buffer_b=300,
            warm_start_strategy="S1",
            adwin_delta=0.02,
            adwin_clock=16,
            use_pruned_arf=True,
            seed=cfg["project"]["seed"]
        ),
        CGTFixedScheduleAblation(seed=cfg["project"]["seed"]),
        ConditionalGatedTrainingController(
            stability_k=cgt_cfg["stability_k"],
            buffer_b=cgt_cfg["buffer_b"],
            warm_start_strategy="S2",
            adwin_delta=cgt_cfg["adwin_delta"],
            adwin_clock=cgt_cfg["adwin_clock"],
            seed=cfg["project"]["seed"]
        ),
        ConditionalGatedTrainingController(
            stability_k=cgt_cfg["stability_k"],
            buffer_b=cgt_cfg["buffer_b"],
            warm_start_strategy="S3",
            adwin_delta=cgt_cfg["adwin_delta"],
            adwin_clock=cgt_cfg["adwin_clock"],
            seed=cfg["project"]["seed"]
        )
    ]

    summaries = []
    all_step_dfs = {}
    
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    for m in models:
        m_name = getattr(m, 'name', m.__class__.__name__)
        print(f"Evaluating {m_name}...")
        summary, step_df = run_prequential_evaluation(m, test_stream)
        summaries.append(summary)
        all_step_dfs[m_name] = step_df
        
        # Save raw step metrics
        safe_name = m_name.replace(":", "").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").lower()
        step_df.to_csv(f"results/raw/steps_{safe_name}.csv", index=False)
        print(f"  -> Accuracy: {summary['accuracy_pct']}%, CPU: {summary['mean_cpu_us_per_instance']} us/inst, Peak Mem: {summary['peak_memory_mb']} MB")

    # Compute CPU savings vs Always-ARF
    summary_df = pd.DataFrame(summaries)
    arf_row = summary_df[summary_df['model_name'] == 'Always-ARF'].iloc[0]
    arf_cpu = arf_row['mean_cpu_us_per_instance']
    
    summary_df['cpu_saving_pct'] = summary_df['mean_cpu_us_per_instance'].apply(
        lambda c: round((1.0 - (c / arf_cpu)) * 100.0, 2)
    )
    
    # Save benchmark summary table
    summary_csv = "results/tables/benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nBenchmark completed successfully. Summary saved to {summary_csv}")
    print("\nBenchmark Summary Table:")
    print(summary_df[['model_name', 'accuracy_pct', 'mean_cpu_us_per_instance', 'cpu_saving_pct', 'peak_memory_mb']])
    
    return summary_df

if __name__ == "__main__":
    run_benchmark()
