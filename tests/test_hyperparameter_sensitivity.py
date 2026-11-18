import pytest
import numpy as np
import pandas as pd
from src.analysis.hyperparameter_sensitivity import (
    compute_synthetic_sensitivity_surface,
    compute_parameter_sensitivity_indices,
)


def test_sensitivity_surface_dimensions_and_types():
    df = compute_synthetic_sensitivity_surface()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 130  # 13 tau x 10 K
    expected_cols = [
        "meta_threshold_tau",
        "cooldown_k",
        "accuracy_pct",
        "mean_cpu_latency_us",
        "duty_cycle_rho_pct",
        "efficiency_rating",
        "sub_1ms_compliant",
    ]
    for col in expected_cols:
        assert col in df.columns
    assert df["accuracy_pct"].isna().sum() == 0
    assert (df["accuracy_pct"] >= 80.0).all()
    assert (df["duty_cycle_rho_pct"] > 0).all()


def test_canonical_operating_point_compliance():
    df = compute_synthetic_sensitivity_surface()
    # Operating point: tau=0.50, K=100
    opt = df[(df["meta_threshold_tau"] == 0.50) & (df["cooldown_k"] == 100)].iloc[0]
    assert opt["sub_1ms_compliant"] is True or opt["sub_1ms_compliant"] == 1
    assert opt["mean_cpu_latency_us"] < 1000.0
    assert opt["efficiency_rating"] >= 8.5
    assert 85.0 <= opt["accuracy_pct"] <= 86.5


def test_monotonic_duty_behavior():
    df = compute_synthetic_sensitivity_surface()
    # At fixed K=100, increasing tau should decrease duty cycle rho
    fixed_k = df[df["cooldown_k"] == 100].sort_values("meta_threshold_tau")
    rhos = fixed_k["duty_cycle_rho_pct"].values
    # Check general downward trend
    assert rhos[0] > rhos[-1], "Duty cycle should be higher at low tau than at high tau"


def test_sensitivity_indices_curvature():
    df = compute_synthetic_sensitivity_surface()
    summary = compute_parameter_sensitivity_indices(df)
    assert summary["operating_tau"] == 0.50
    assert summary["operating_k"] == 100
    assert summary["curvature_status"] == "CONVEX_STABLE_BASIN"
    assert abs(summary["d_acc_d_tau"]) < 2.0  # Stable gradient
    assert summary["basin_width_tau_plus_minus"] >= 0.10
