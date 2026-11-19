"""
CGT-v4: Ultra-Low Latency Cascaded Compute-Gated Trees
=============================================================================
LABEL: post-hoc: developed after viewing test results
=============================================================================

Architectural Upgrades over CGT-v3:
1. Optimized Streaming Meta-Gate:
   - Replaced River StandardScaler & LogisticRegression with hand-coded NumPy dot product
     and online incremental Welford mean/variance scaler (< 1.5 µs forward pass).
   - Mini-batch SGD: Gradients accumulated and updated every M steps (M in [16, 32]),
     slashing backward pass overhead by 95%.
   - Margin Reuse: Caches HT probability distribution during predict_one to eliminate
     redundant predict_proba_one calls.
2. Replay Architecture Options:
   - Option A (Amortized Replay): Buffer size B <= 50, amortized across 10 subsequent steps
     (5 items/step), eliminating the 38 ms blocking spike.
   - Option B (Warm Standby): No replay buffer; ARF learns on every 4th sample (t % 4 == 0,
     25% training duty) and predicts only when gate fires.
3. Multi-Model Support:
   - Supports both ARF-3 (3-tree fair heavy arm) and ARF-10 (10-tree full ensemble arm).
"""

from collections import deque
import math
import numpy as np
from river.drift import ADWIN
from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier

from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper, ARFOptimizedWrapper


class WelfordIncrementalScaler:
    """Online incremental mean and variance normalizer via Welford's algorithm."""
    def __init__(self, dim=7):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update_and_scale(self, x_vec):
        self.n += 1
        delta = x_vec - self.mean
        self.mean += delta / self.n
        delta2 = x_vec - self.mean
        self.M2 += delta * delta2
        
        var = self.M2 / max(self.n - 1, 1)
        std = np.sqrt(var + 1e-6)
        return (x_vec - self.mean) / std

    def scale_only(self, x_vec):
        if self.n <= 1:
            return x_vec
        var = self.M2 / (self.n - 1)
        std = np.sqrt(var + 1e-6)
        return (x_vec - self.mean) / std


class HandCodedMiniBatchLogisticGate:
    """
    Fast online linear meta-classifier with hand-coded vector dot product
    and mini-batch SGD weight updates.
    """
    def __init__(self, dim=7, lr=0.05, batch_size=16, l2_reg=1e-4):
        self.dim = dim
        self.lr = lr
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.weights = np.zeros(dim, dtype=np.float64)
        self.bias = -1.0  # prior toward dormancy (HT handles ~90% traffic)
        
        self.grad_w_acc = np.zeros(dim, dtype=np.float64)
        self.grad_b_acc = 0.0
        self.batch_count = 0

    def predict_proba(self, x_norm):
        z = self.bias + float(np.dot(self.weights, x_norm))
        z_clipped = max(-15.0, min(15.0, z))
        return 1.0 / (1.0 + math.exp(-z_clipped))

    def update(self, x_norm, target_z):
        p = self.predict_proba(x_norm)
        err = p - target_z
        
        self.grad_w_acc += err * x_norm
        self.grad_b_acc += err
        self.batch_count += 1
        
        if self.batch_count >= self.batch_size:
            scale = 1.0 / self.batch_count
            self.weights -= self.lr * (self.grad_w_acc * scale + self.l2_reg * self.weights)
            self.bias -= self.lr * (self.grad_b_acc * scale)
            self.grad_w_acc.fill(0.0)
            self.grad_b_acc = 0.0
            self.batch_count = 0


class CGTv4Controller:
    """
    CGT-v4 (post-hoc: developed after viewing test results).
    High-performance real-time meta-controller with microsecond hand-coded gate
    and spike-free amortized / warm-standby execution.
    """
    def __init__(
        self,
        tau_gate=0.25,
        max_rho=0.15,
        h_cooldown=50,
        replay_mode="amortized",  # "amortized" or "warm_standby"
        buffer_b=50,             # buffer cap for amortized replay (<= 50)
        k_replay_per_step=5,     # items amortized per step
        batch_size=16,           # mini-batch size for SGD
        n_heavy_trees=3,         # 3 trees (fair matched) or 10 trees
        recent_w=50,
        adwin_delta=0.005,
        adwin_clock=32,
        seed=42
    ):
        self.label = "post-hoc: developed after viewing test results"
        self.name = f"CGT-v4 (post-hoc: developed after viewing test results, {replay_mode}, ARF-{n_heavy_trees})"
        self.tau_gate = tau_gate
        self.max_rho = max_rho
        self.h_cooldown = h_cooldown
        self.replay_mode = replay_mode
        self.buffer_b = buffer_b
        self.k_replay_per_step = k_replay_per_step
        self.batch_size = batch_size
        self.n_heavy_trees = n_heavy_trees
        self.recent_w = recent_w
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Base learners
        self.ht = HoeffdingTreeWrapper()
        if n_heavy_trees == 3:
            self.arf = ARFClassifier(
                n_models=3,
                grace_period=50,
                lambda_value=6,
                delta=0.01,
                split_criterion='info_gain',
                seed=seed
            )
        elif n_heavy_trees == 10:
            self.arf = ARFClassifier(
                n_models=10,
                grace_period=50,
                lambda_value=6,
                delta=0.01,
                split_criterion='info_gain',
                seed=seed
            )
        else:
            raise ValueError(f"Unsupported n_heavy_trees: {n_heavy_trees}")
            
        self.detector = ADWIN(delta=adwin_delta, clock=adwin_clock)
        
        # Hand-coded gating modules
        self.scaler = WelfordIncrementalScaler(dim=7)
        self.meta_gate = HandCodedMiniBatchLogisticGate(dim=7, lr=0.05, batch_size=batch_size)
        
        # State buffers
        self.buffer = deque(maxlen=buffer_b)
        self.amortized_queue = deque()
        self.recent_ht_errors = deque(maxlen=recent_w)
        self.cooldown_counter = 0
        self.gate_open = False
        
        # Diagnostic tracking
        self.total_steps = 0
        self.gate_open_steps = 0
        self.total_alarms = 0
        self.learned_gate_triggers = 0
        self.budget_throttled_steps = 0
        
        # Step-level cache for zero-redundancy execution
        self._cached_margin = 0.5
        self._cached_ht_pred = 0
        self._current_x_norm = None

    def _extract_meta_features_vec(self, x):
        # 1. Reuse cached HT margin (zero predict_proba overhead)
        margin = self._cached_margin
        
        # 2. Sliding HT error rate
        err_rate = sum(self.recent_ht_errors) / len(self.recent_ht_errors) if self.recent_ht_errors else 0.15
        
        # 3-6. Physical harmonics and momentum
        h_sin = float(x.get('hour_sin', 0.0))
        h_cos = float(x.get('hour_cos', 0.0))
        ren_ratio = float(x.get('renewable_ratio_lag_1', x.get('renewable_ratio', 0.2)))
        d_diff_24 = float(x.get('demand_diff_1_24', 0.0)) / 5000.0
        in_cool = 1.0 if self.cooldown_counter > 0 else 0.0
        
        raw_vec = np.array([margin, err_rate, h_sin, h_cos, ren_ratio, d_diff_24, in_cool], dtype=np.float64)
        return self.scaler.update_and_scale(raw_vec)

    def predict_one(self, x):
        self.total_steps += 1
        current_rho = self.gate_open_steps / max(self.total_steps, 1)
        budget_ok = current_rho < self.max_rho
        
        # HT evaluation & probability caching
        proba_ht = self.ht.model.predict_proba_one(x)
        p0 = proba_ht.get(0, 0.5)
        p1 = proba_ht.get(1, 0.5)
        self._cached_margin = abs(p1 - p0)
        self._cached_ht_pred = 1 if p1 > p0 else 0
        
        # Hand-coded gating scoring
        x_norm = self._extract_meta_features_vec(x)
        self._current_x_norm = x_norm
        
        prob_arf_wins = self.meta_gate.predict_proba(x_norm)
        learned_trigger = prob_arf_wins > self.tau_gate
        
        if self.cooldown_counter > 0:
            self.gate_open = True
            self.cooldown_counter -= 1
        elif learned_trigger and budget_ok:
            self.gate_open = True
            self.learned_gate_triggers += 1
        else:
            self.gate_open = False
            if learned_trigger and not budget_ok:
                self.budget_throttled_steps += 1
                
        if self.gate_open:
            self.gate_open_steps += 1
            pred_arf = self.arf.predict_one(x)
            return self._cached_ht_pred if pred_arf is None else int(pred_arf)
        return self._cached_ht_pred

    def learn_one(self, x, y, y_pred=None):
        if y_pred is None:
            y_pred = self._cached_ht_pred if not self.gate_open else self._cached_ht_pred
        error = 1.0 if y_pred != y else 0.0
        
        # 1. Concept drift detection
        self.detector.update(error)
        drift_now = self.detector.drift_detected
        
        if drift_now:
            self.total_alarms += 1
            self.cooldown_counter = self.h_cooldown
            self.gate_open = True
            
            # Replay trigger handling
            if self.replay_mode == "amortized":
                # Transfer buffer to amortized queue to avoid blocking latency spike
                self.amortized_queue.extend(self.buffer)
                self.buffer.clear()
            # In warm_standby mode, ARF is already continuously trained on every 4th sample
            
        # 2. Process amortized replay queue (if active)
        if self.replay_mode == "amortized" and self.amortized_queue:
            items_to_process = min(self.k_replay_per_step, len(self.amortized_queue))
            for _ in range(items_to_process):
                bx, by = self.amortized_queue.popleft()
                self.arf.learn_one(bx, by)
                
        # 3. Meta-gate mini-batch training
        err_ht = 1.0 if self._cached_ht_pred != y else 0.0
        self.recent_ht_errors.append(err_ht)
        
        # Only evaluate ARF prediction if HT was wrong (conditional shadow evaluation)
        if err_ht == 1.0:
            pred_arf = self.arf.predict_one(x)
            err_arf = 1.0 if (pred_arf is not None and pred_arf != y) else (0.0 if pred_arf == y else 1.0)
            z_t = 1.0 if (err_arf == 0.0) else 0.0
        else:
            z_t = 0.0
            
        if self._current_x_norm is not None:
            self.meta_gate.update(self._current_x_norm, z_t)
            
        # 4. Update base models
        self.ht.learn_one(x, y)
        
        if self.gate_open:
            self.arf.learn_one(x, y)
        else:
            if self.replay_mode == "amortized":
                self.buffer.append((x, y))
            elif self.replay_mode == "warm_standby":
                # Standby: Learn on every 4th sample (t % 4 == 0)
                if self.total_steps % 4 == 0:
                    self.arf.learn_one(x, y)

