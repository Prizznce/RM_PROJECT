import time
import os
import psutil
from collections import deque
import numpy as np
import pandas as pd

def run_prequential_evaluation(model, stream, window_size=500, log_every=100):
    """
    Standard prequential (test-then-train) evaluation loop.
    Strictly follows Gama et al. (2013) prequential protocol.
    
    Returns:
      results_dict: Summary metrics (accuracy, CPU time, peak memory, etc.)
      step_logs: List of sliding window metrics across the stream for plotting
    """
    process = psutil.Process()
    initial_mem_mb = process.memory_info().rss / (1024 * 1024)
    peak_mem_mb = initial_mem_mb
    
    total_instances = 0
    correct_count = 0
    total_cpu_ns = 0
    
    sliding_window = deque(maxlen=window_size)
    sliding_acc_history = []
    step_indices = []
    timestamps = []
    gate_states = []
    
    for x_t, y_t, ts in stream:
        total_instances += 1
        
        # 1. Measure instance processing time (inference + decision + learning)
        t_start = time.perf_counter_ns()
        
        # Prequential Step 1: Predict before label is revealed
        y_pred = model.predict_one(x_t)
        
        # Evaluate
        is_correct = int(y_pred == y_t)
        correct_count += is_correct
        sliding_window.append(is_correct)
        
        # Prequential Step 2: Learn from revealed label
        if hasattr(model, 'learn_one'):
            try:
                # Controllers accept y_pred for internal monitoring
                model.learn_one(x_t, y_t, y_pred)
            except TypeError:
                # Base learners take (x_t, y_t)
                model.learn_one(x_t, y_t)
                
        t_elapsed = time.perf_counter_ns() - t_start
        total_cpu_ns += t_elapsed
        
        # Track gate state if controller exposes it
        current_gate = 1 if getattr(model, 'gate_open', False) else (1 if getattr(model, 'active_model', '') == 'ARF' else 0)
        
        # Periodic memory and step logging
        if total_instances % log_every == 0 or total_instances == len(stream):
            current_mem = process.memory_info().rss / (1024 * 1024)
            if current_mem > peak_mem_mb:
                peak_mem_mb = current_mem
                
            sliding_acc = (sum(sliding_window) / len(sliding_window)) * 100.0
            sliding_acc_history.append(sliding_acc)
            step_indices.append(total_instances)
            timestamps.append(str(ts))
            gate_states.append(current_gate)

    overall_acc = (correct_count / total_instances) * 100.0
    mean_cpu_us = (total_cpu_ns / total_instances) / 1000.0 # microseconds per instance
    
    summary = {
        'model_name': getattr(model, 'name', model.__class__.__name__),
        'total_instances': total_instances,
        'accuracy_pct': round(overall_acc, 2),
        'mean_cpu_us_per_instance': round(mean_cpu_us, 2),
        'total_cpu_sec': round(total_cpu_ns / 1e9, 3),
        'peak_memory_mb': round(peak_mem_mb, 2),
        'gate_open_fraction': round(float(getattr(model, 'gate_open_steps', 0) / max(total_instances, 1)), 4) if hasattr(model, 'gate_open_steps') else None,
        'total_alarms': int(getattr(model, 'total_alarms', 0)) if hasattr(model, 'total_alarms') else None,
        'replay_instances': int(getattr(model, 'replay_instances_trained', 0)) if hasattr(model, 'replay_instances_trained') else None
    }
    
    step_df = pd.DataFrame({
        'step': step_indices,
        'timestamp': timestamps,
        'sliding_acc': sliding_acc_history,
        'gate_open': gate_states
    })
    
    return summary, step_df
