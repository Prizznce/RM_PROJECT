import pytest
import numpy as np
from src.controllers.oracle_switch import OracleGatedSwitch
from src.models.stronger_baselines import (
    NaivePersistenceHourly,
    NaivePersistenceDiurnal,
    ZeroSkillMajorityClass
)

def test_persistence_baselines():
    p_hourly = NaivePersistenceHourly()
    p_diurnal = NaivePersistenceDiurnal()
    majority = ZeroSkillMajorityClass()

    x_up = {'demand_diff_1_2': 150.0, 'demand_diff_1_24': 300.0}
    x_down = {'demand_diff_1_2': -80.0, 'demand_diff_1_24': -200.0}
    x_zero = {'demand_diff_1_2': 0.0, 'demand_diff_1_24': 0.0}

    assert p_hourly.predict_one(x_up) == 1
    assert p_hourly.predict_one(x_down) == 0
    assert p_hourly.predict_one(x_zero) == 0

    assert p_diurnal.predict_one(x_up) == 1
    assert p_diurnal.predict_one(x_down) == 0
    assert p_diurnal.predict_one(x_zero) == 0

    assert majority.predict_one(x_up) == 0
    assert majority.predict_one(x_down) == 0

def test_oracle_switch_logic():
    oracle = OracleGatedSwitch(seed=42)
    
    # Mock HT and ARF predictions by setting up mock instances or monkey-patching
    class MockLearner:
        def __init__(self, pred):
            self.pred = pred
        def predict_one(self, x):
            return self.pred
        def learn_one(self, x, y):
            pass

    # Case 1: HT correct, ARF correct -> Oracle selects HT (cheapest)
    oracle.ht = MockLearner(1)
    oracle.arf = MockLearner(1)
    y_pred, arm = oracle.evaluate_and_learn({'dummy': 1}, 1)
    assert y_pred == 1
    assert arm == "HT"
    assert oracle.gate_open is False

    # Case 2: HT wrong, ARF correct -> Oracle selects ARF (essential win)
    oracle.ht = MockLearner(0)
    oracle.arf = MockLearner(1)
    y_pred, arm = oracle.evaluate_and_learn({'dummy': 1}, 1)
    assert y_pred == 1
    assert arm == "ARF"
    assert oracle.gate_open is True

    # Case 3: HT wrong, ARF wrong -> Oracle selects HT (avoid wasted compute)
    oracle.ht = MockLearner(0)
    oracle.arf = MockLearner(0)
    y_pred, arm = oracle.evaluate_and_learn({'dummy': 1}, 1)
    assert y_pred == 0
    assert arm == "HT"
    assert oracle.gate_open is False

    # Case 4: HT correct, ARF wrong -> Oracle selects HT
    oracle.ht = MockLearner(1)
    oracle.arf = MockLearner(0)
    y_pred, arm = oracle.evaluate_and_learn({'dummy': 1}, 1)
    assert y_pred == 1
    assert arm == "HT"
    assert oracle.gate_open is False

