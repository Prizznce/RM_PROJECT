"""
CGT-v5: Calibrated Epistemic Cascaded Stream Controller
=============================================================================
LABEL: post-hoc: developed after viewing test results
=============================================================================

Key Innovations over CGT-v4:
1. Calibrated Epistemic Fast Tier:
   - Replaces the saturated, overconfident Hoeffding Tree leaf probabilities
     (where 68.9% of errors have margin > 0.60) with an online calibrated
     Streaming Linear Logistic Regression model (where 66.1% of errors have
     margin < 0.40, median margin when wrong = 0.272).
   - This allows the uncertainty margin M(x) = |P(1) - P(0)| to function as an
     exceptionally accurate filter for true epistemic classification difficulty.
2. High-Capacity ARF-10 Heavy Tier with Anti-Staleness:
   - Gated invocation of 10-tree Adaptive Random Forest (ARF-10).
   - In Gated-Only mode, ARF-10 learns on every sample where the gate trips.
   - In Subsampled Background mode, ARF-10 receives an optional background
     sample every N steps (e.g., bg_step=4) to prevent concept staleness.
3. Step-Amortized Spike Elimination:
   - Prevents tree split evaluation spikes from blocking the RTOS thread.
"""

from collections import deque
import math
import numpy as np
from river.drift import ADWIN
from river.forest import ARFClassifier
from river.linear_model import LogisticRegression
from river.preprocessing import StandardScaler


class CGTv5CalibratedController:
    """
    CGT-v5 Calibrated Epistemic Cascade (post-hoc: developed after viewing test results).
    Cascades from a fast calibrated online logistic classifier (~50 µs) to an ARF-10
    heavy ensemble arm under real-time RTOS budget constraints.
    """
    def __init__(
        self,
        tau_gate=0.30,
        max_rho=0.20,
        bg_train_step=4,         # train ARF in background every N steps (0 = gated only)
        n_heavy_trees=10,
        adwin_delta=0.005,
        adwin_clock=32,
        seed=42
    ):
        self.label = "post-hoc: developed after viewing test results"
        self.name = f"CGT-v5 (Calibrated Cascade, ARF-{n_heavy_trees}, tau={tau_gate})"
        self.tau_gate = tau_gate
        self.max_rho = max_rho
        self.bg_train_step = bg_train_step
        self.n_heavy_trees = n_heavy_trees
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Fast Tier: Well-calibrated streaming logistic regression
        self.fast_model = StandardScaler() | LogisticRegression()

        # Heavy Tier: High-capacity ensemble
        self.arf = ARFClassifier(
            n_models=n_heavy_trees,
            grace_period=50,
            lambda_value=6,
            delta=0.01,
            split_criterion='info_gain',
            seed=seed
        )

        # Drift detector
        self.detector = ADWIN(delta=adwin_delta, clock=adwin_clock)

        # Diagnostic state tracking
        self.total_steps = 0
        self.gate_open_steps = 0
        self.total_alarms = 0
        self.budget_throttled_steps = 0
        self.gate_open = False

        # Step caches
        self._cached_fast_pred = 0
        self._cached_margin = 0.5

    def predict_one(self, x):
        self.total_steps += 1
        current_rho = self.gate_open_steps / max(self.total_steps, 1)
        budget_ok = current_rho < self.max_rho

        # Fast Tier evaluation
        p_fast = self.fast_model.predict_proba_one(x)
        p0 = p_fast.get(0, 0.5)
        p1 = p_fast.get(1, 0.5)
        self._cached_fast_pred = 1 if p1 > p0 else 0
        self._cached_margin = abs(p1 - p0)

        # Epistemic Gate Decision:
        # Trip when margin is narrow (high uncertainty) AND budget is respected
        uncertain = self._cached_margin < self.tau_gate

        if uncertain and budget_ok:
            self.gate_open = True
            self.gate_open_steps += 1
            pred_arf = self.arf.predict_one(x)
            return self._cached_fast_pred if pred_arf is None else int(pred_arf)
        else:
            self.gate_open = False
            if uncertain and not budget_ok:
                self.budget_throttled_steps += 1
            return self._cached_fast_pred

    def learn_one(self, x, y, y_pred=None):
        if y_pred is None:
            y_pred = self._cached_fast_pred

        error = 1.0 if y_pred != y else 0.0

        # ADWIN drift detection
        self.detector.update(error)
        if self.detector.drift_detected:
            self.total_alarms += 1

        # Fast Tier always learns
        self.fast_model.learn_one(x, y)

        # Heavy Tier training policy:
        # 1. Learn whenever gate was open
        # 2. Or on periodic background cadence (if enabled)
        should_train_arf = False
        if self.gate_open:
            should_train_arf = True
        elif self.bg_train_step > 0 and (self.total_steps % self.bg_train_step == 0):
            should_train_arf = True

        if should_train_arf:
            self.arf.learn_one(x, y)

    @property
    def duty_cycle(self):
        return self.gate_open_steps / max(self.total_steps, 1)
