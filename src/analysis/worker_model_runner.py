"""
Worker script executed in an ISOLATED subprocess for each policy.
Fast & accurate profiling protocol:
  1. Warm-up on 1,000 instances.
  2. Record initial RSS (MB).
  3. Tracemalloc peak memory (MB) evaluated over 1,000 streaming steps.
  4. RSS delta (MB).
  5. 5 independent latency passes over 1,000 instances (warm-up excluded).
Outputs JSON to stdout.
"""

import sys
import json
import time
import tracemalloc
import psutil
import os
import yaml
import numpy as np

from src.data.stream_loader import ElectricityStreamLoader
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper, ARFOptimizedWrapper
from src.models.stronger_baselines import (
    ZeroSkillMajorityClass,
    NaivePersistenceHourly,
    StreamingSGDClassifier,
    AdaptiveHoeffdingTreeWrapper,
    StreamingKNNWrapper,
    LeveragingBaggingWrapper,
    StreamingRandomPatchesWrapper
)
from src.controllers.a1_threshold import A1ThresholdController
from src.controllers.a2_ucb import A2UCBController
from src.analysis.run_day2_benchmarks import A4FixedRouter
from src.controllers.oms_mab import OMSMABController
from src.controllers.cgt import ConditionalGatedTrainingController
from src.controllers.cgt_v2 import CGTv2Controller
from src.controllers.cgt_v3 import CGTv3LearnedGateController
from src.controllers.oracle_switch import OracleGatedSwitch

def get_policy_instance(model_key, seed=42):
    with open("configs/frozen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if model_key == "majority":
        return ZeroSkillMajorityClass()
    elif model_key == "persistence":
        return NaivePersistenceHourly()
    elif model_key == "sgd":
        return StreamingSGDClassifier(lr=0.01)
    elif model_key == "knn":
        return StreamingKNNWrapper(n_neighbors=5, window_size=100)
    elif model_key == "always_ht":
        return HoeffdingTreeWrapper()
    elif model_key == "hat":
        return AdaptiveHoeffdingTreeWrapper(grace_period=200, seed=seed)
    elif model_key == "levbag":
        return LeveragingBaggingWrapper(n_models=3, grace_period=100, seed=seed)
    elif model_key == "srp":
        return StreamingRandomPatchesWrapper(n_models=3, grace_period=100, seed=seed)
    elif model_key == "always_arf":
        return ARFFullWrapper(seed=seed)
    elif model_key == "a1":
        return A1ThresholdController(theta_hi=0.25, theta_lo=0.15, window_size=50, seed=seed)
    elif model_key == "a2":
        return A2UCBController(c=0.5, discount=0.98, seed=seed)
    elif model_key == "a3":
        return ARFOptimizedWrapper(n_models=5, seed=seed)
    elif model_key == "a4":
        return A4FixedRouter(batch_size=50, lambda_c=0.02, epsilon=0.05, seed=seed)
    elif model_key == "oms_mab":
        return OMSMABController(gamma=0.98, c=0.5, seed=seed)
    elif model_key == "cgt_v1":
        return ConditionalGatedTrainingController(
            stability_k=cfg["cgt_frozen"]["stability_k"],
            buffer_b=cfg["cgt_frozen"]["buffer_b"],
            warm_start_strategy="S1",
            adwin_delta=cfg["cgt_frozen"]["adwin_delta"],
            adwin_clock=cfg["cgt_frozen"]["adwin_clock"],
            seed=seed
        )
    elif model_key == "cgt_v2":
        return CGTv2Controller(
            theta_margin=0.25,
            max_rho=0.15,
            h_cooldown=100,
            buffer_b=200,
            use_pruned_arf=True,
            seed=seed
        )
    elif model_key == "cgt_v3":
        return CGTv3LearnedGateController(
            tau_gate=0.20,
            max_rho=0.10,
            h_cooldown=50,
            buffer_b=200,
            use_pruned_arf=True,
            seed=seed
        )
    elif model_key == "oracle":
        return OracleGatedSwitch(seed=seed)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

def run_isolated_benchmark(model_key):
    proc = psutil.Process(os.getpid())
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    full_stream = list(loader.iter_stream(warmup_rows=24))

    val_len = 10512
    # Standardized evaluation slice from test stream
    warmup_slice = full_stream[val_len:val_len + 1000]
    eval_slice = full_stream[val_len + 1000:val_len + 2000]

    model = get_policy_instance(model_key)

    # 1. Warm-up on 1,000 instances
    for x, y, ts in warmup_slice:
        y_pred = model.predict_one(x)
        try:
            model.learn_one(x, y, y_pred)
        except TypeError:
            model.learn_one(x, y)

    # 2. Record initial RSS after warm-up
    rss_initial_mb = proc.memory_info().rss / (1024 * 1024)

    # 3. Tracemalloc over 1,000 evaluation steps
    tracemalloc.start()
    for x, y, ts in eval_slice:
        y_pred = model.predict_one(x)
        try:
            model.learn_one(x, y, y_pred)
        except TypeError:
            model.learn_one(x, y)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_tracemalloc_mb = peak_bytes / (1024 * 1024)
    rss_final_mb = proc.memory_info().rss / (1024 * 1024)
    rss_delta_mb = max(rss_final_mb - rss_initial_mb, 0.0)

    # 4. Measure 5 independent latency passes over 1,000 instances
    pass_latencies_us = []
    for trial in range(5):
        bench_model = get_policy_instance(model_key)
        # 200 warm-up steps
        for x, y, ts in eval_slice[:200]:
            p = bench_model.predict_one(x)
            try:
                bench_model.learn_one(x, y, p)
            except TypeError:
                bench_model.learn_one(x, y)

        t_start = time.perf_counter_ns()
        for x, y, ts in eval_slice[200:]:
            p = bench_model.predict_one(x)
            try:
                bench_model.learn_one(x, y, p)
            except TypeError:
                bench_model.learn_one(x, y)
        t_elapsed_ns = time.perf_counter_ns() - t_start
        mean_us = (t_elapsed_ns / 800) / 1000.0
        pass_latencies_us.append(mean_us)

    lat_mean = float(np.mean(pass_latencies_us))
    lat_std = float(np.std(pass_latencies_us))

    output = {
        "model_key": model_key,
        "tracemalloc_peak_mb": round(peak_tracemalloc_mb, 2),
        "rss_delta_mb": round(rss_delta_mb, 2),
        "latency_mean_us": round(lat_mean, 2),
        "latency_std_us": round(lat_std, 2)
    }

    print("RESULT_JSON:" + json.dumps(output), flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_isolated_benchmark(sys.argv[1])
