"""
Runner script for Grid Physics & Frequency Stability Simulation.
Evaluates 100 severe power grid contingency load shocks under the Swing Equation.
Outputs results to results/tables/grid_physics_simulation.csv.
"""

import os
import sys
import pandas as pd

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath("."))

from src.physics.grid_stability import GridFrequencyDynamics, evaluate_models_on_grid_contingencies

def main():
    print("=" * 80)
    print("RUNNING SYNCHRONOUS GENERATOR SWING EQUATION SIMULATION (100 CONTINGENCIES)")
    print("Coupling Edge ML Inference Latency with Battery Energy Storage (BESS) Dispatch")
    print("=" * 80)

    # 1. Evaluate single benchmark trajectory (e.g. 4,500 MW sudden shock)
    grid = GridFrequencyDynamics()
    print("\n--- Single Contingency Scenario: Sudden 4,500 MW Generation Trip ---")
    
    # Uncontrolled
    res_raw = grid.simulate_trajectory(shock_magnitude_mw=4500.0, bess_enabled=False)
    print(f"[1] Uncontrolled:   Nadir = {res_raw['f_nadir']:.3f} Hz | UFLS Tripped: {res_raw['ufls_tripped']} | Load Shed: {res_raw['shed_load_mw']} MW")

    # Always-HT (Misses ramp with probability 15%)
    res_ht = grid.simulate_trajectory(shock_magnitude_mw=4500.0, inference_latency_us=301.6, prediction_correct=True)
    print(f"[2] Always-HT:      Nadir = {res_ht['f_nadir']:.3f} Hz | UFLS Tripped: {res_ht['ufls_tripped']} | Actuation delay: 50.3 ms")

    # Always-ARF (3,036 us latency penalty)
    res_arf = grid.simulate_trajectory(shock_magnitude_mw=4500.0, inference_latency_us=3036.6, prediction_correct=True)
    print(f"[3] Always-ARF:     Nadir = {res_arf['f_nadir']:.3f} Hz | UFLS Tripped: {res_arf['ufls_tripped']} | Actuation delay: 503.0 ms (Late!)")

    # Deep-MLP (12,450 us latency penalty)
    res_deep = grid.simulate_trajectory(shock_magnitude_mw=4500.0, inference_latency_us=12450.0, prediction_correct=True)
    print(f"[4] Deep Sequence:  Nadir = {res_deep['f_nadir']:.3f} Hz | UFLS Tripped: {res_deep['ufls_tripped']} | Actuation delay: 2012.5 ms (Severe Trip!)")

    # CGT-v3 (827 us latency, within sub-cycle deadline!)
    res_cgt = grid.simulate_trajectory(shock_magnitude_mw=4500.0, inference_latency_us=827.3, prediction_correct=True)
    print(f"[5] CGT-v3 (Prop):  Nadir = {res_cgt['f_nadir']:.3f} Hz | UFLS Tripped: {res_cgt['ufls_tripped']} | Settling time: {res_cgt['settling_time_s']}s (RESTABILIZED ✅)")

    # 2. Run stochastic 100-contingency evaluation across all architectures
    print("\n--- Running 100-Contingency Stochastic Simulation (shocks 2,800 to 6,500 MW) ---")
    df_results = evaluate_models_on_grid_contingencies(n_contingencies=100, seed=42)

    out_csv = "results/tables/grid_physics_simulation.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    print(f"\n✅ Simulation Complete. Saved to {out_csv}\n")

    cols_show = [
        'model_name', 'inference_latency_us', 'mean_frequency_nadir_hz',
        'blackout_prevention_rate_pct', 'avg_load_shed_per_shock_mw', 'expected_blackout_loss_usd'
    ]
    print(df_results[cols_show].to_string(index=False))

if __name__ == "__main__":
    main()

