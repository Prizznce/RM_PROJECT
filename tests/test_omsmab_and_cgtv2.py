import pytest
import numpy as np
from src.controllers.oms_mab import OMSMABController
from src.controllers.cgt_v2 import CGTv2Controller

def test_oms_mab_initialization_and_discounting():
    oms = OMSMABController(gamma=0.95, c=0.5, seed=42)
    assert len(oms.arms) == 2
    assert oms.k == 2
    
    # Process 10 dummy instances
    for i in range(10):
        x = {'demand_lag_1': 20000.0 + i * 10, 'demand_diff_1_2': 10.0}
        pred = oms.predict_one(x)
        assert pred in (0, 1)
        oms.learn_one(x, y=1, y_pred=pred)
        
    assert oms.total_steps == 10
    assert np.all(oms.discounted_pulls > 0)
    assert np.all(oms.arm_pull_counts > 0)

def test_cgt_v2_budget_enforcement():
    # Test with very strict budget: max_rho = 0.10
    ctrl = CGTv2Controller(
        theta_margin=0.99, # artificially high so it is always uncertain
        max_rho=0.10,
        h_cooldown=0,
        seed=42
    )
    
    # Run 100 steps
    for i in range(100):
        x = {'demand_lag_1': 25000.0, 'demand_diff_1_2': -20.0}
        pred = ctrl.predict_one(x)
        ctrl.learn_one(x, y=0, y_pred=pred)
        
    final_rho = ctrl.gate_open_steps / ctrl.total_steps
    # Budget must strictly constrain gate_open fraction to <= max_rho + slight tolerance for step 1
    assert final_rho <= 0.12, f"Budget exceeded: expected <= 0.12, got {final_rho}"
    assert ctrl.budget_throttled_steps > 50, "Uncertainty triggers should have been throttled by budget"

def test_cgt_v2_cooldown():
    ctrl = CGTv2Controller(
        theta_margin=0.0, # no uncertainty triggers
        max_rho=0.50,
        h_cooldown=5,
        seed=42
    )
    
    # Simulate a drift trigger by setting cooldown_counter directly
    ctrl.cooldown_counter = 5
    for i in range(5):
        x = {'demand_lag_1': 25000.0}
        ctrl.predict_one(x)
        assert ctrl.gate_open is True, f"Gate should be open during cooldown step {i+1}"
        ctrl.learn_one(x, y=1, y_pred=1)
        
    # Step 6: Cooldown expired
    x = {'demand_lag_1': 25000.0}
    ctrl.predict_one(x)
    assert ctrl.gate_open is False, "Gate should be closed after cooldown expires"

