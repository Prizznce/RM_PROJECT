import pytest
import numpy as np
import pandas as pd
from src.data.stream_loader import ElectricityStreamLoader

def test_causal_feature_leakage():
    """
    Automated test verifying zero future leakage.
    Features x_t must remain invariant to modifications of future data (t, t+1, ...).
    """
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    stream = list(loader.iter_stream(warmup_rows=24))

    assert len(stream) == 35040, f"Stream must contain exactly 35,040 instances, got {len(stream)}."

    # Check a specific step t = 100
    t_idx = 100
    x_t, y_t, ts = stream[t_idx]

    # Verify that x_t matches exactly the historical values in df at t-1, t-2, t-24
    actual_df_idx = t_idx + 24
    df = loader.df

    assert x_t['demand_lag_1'] == df.loc[actual_df_idx - 1, 'Demand (MW)']
    assert x_t['demand_lag_2'] == df.loc[actual_df_idx - 2, 'Demand (MW)']
    assert x_t['demand_lag_24'] == df.loc[actual_df_idx - 24, 'Demand (MW)']
    assert x_t['solar_lag_1'] == df.loc[actual_df_idx - 1, 'Solar (MW)']
    assert x_t['wind_lag_1'] == df.loc[actual_df_idx - 1, 'Wind (MW)']
    assert x_t['gen_lag_1'] == df.loc[actual_df_idx - 1, 'Total Generation (MW)']
    assert x_t['demand_diff_1_24'] == df.loc[actual_df_idx - 1, 'Demand (MW)'] - df.loc[actual_df_idx - 24, 'Demand (MW)']

    if 'Nuclear (MW)' in df.columns:
        assert x_t['nuclear_lag_1'] == df.loc[actual_df_idx - 1, 'Nuclear (MW)']
    if 'Gas (MW)' in df.columns:
        assert x_t['gas_lag_1'] == df.loc[actual_df_idx - 1, 'Gas (MW)']

    # Verify that changing future demand does not affect x_t
    original_demand_t = df.loc[actual_df_idx, 'Demand (MW)']
    try:
        # Simulate perturbation of future demand
        df.loc[actual_df_idx, 'Demand (MW)'] = 9999999.0
        pert_stream = list(loader.iter_stream(warmup_rows=24))
        x_t_pert, _, _ = pert_stream[t_idx]
        
        # x_t must NOT change
        assert x_t == x_t_pert, "Leakage detected! x_t changed when Demand_t was modified."
    finally:
        df.loc[actual_df_idx, 'Demand (MW)'] = original_demand_t

def test_target_definition():
    """
    Verify that target y_t is strictly binary and correctly computed from Demand_t vs Demand_{t-1}.
    """
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    df = loader.df

    for i, (x_t, y_t, ts) in enumerate(loader.iter_stream(warmup_rows=24)):
        actual_idx = i + 24
        expected_y = int(df.loc[actual_idx, 'Demand (MW)'] > df.loc[actual_idx - 1, 'Demand (MW)'])
        assert y_t == expected_y, f"Target mismatch at index {actual_idx}: expected {expected_y}, got {y_t}"
        assert y_t in (0, 1), f"Target {y_t} is not binary."
        if i >= 100:  # test first 100 instances
            break
