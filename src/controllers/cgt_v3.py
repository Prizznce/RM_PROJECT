from collections import deque
import math
import numpy as np
from river.drift import ADWIN
from river.linear_model import LogisticRegression
from river.preprocessing import StandardScaler
from src.models.base_models import HoeffdingTreeWrapper, ARFOptimizedWrapper, ARFFullWrapper

class CGTv3LearnedGateController:
    """
    CGT-v3: Learned Meta-Gate Controller.
    
    A lightweight online linear meta-classifier predicts whether ARF would beat HT
    on each incoming streaming instance, subject to a hard computational budget cap (rho <= max_rho)
    and an ADWIN concept-drift cooldown interrupt.
    
    Meta-Features s_t:
      1. ht_margin: |P(y=1|x) - P(y=0|x)| (HT probability margin / confidence)
      2. recent_error_ht: sliding error rate of HT over last W instances
      3. hour_sin: sin(2 * pi * hour / 24)
      4. hour_cos: cos(2 * pi * hour / 24)
      5. renewable_ratio: (Solar + Wind) / TotalGen
      6. demand_diff_1_24: diurnal 24-hr load momentum (normalized)
      7. in_cooldown: binary indicator if system is in post-drift recovery
      
    Training Target for Meta-Gate z_t:
      z_t = 1 if (ARF is correct AND HT is wrong) else 0.
      This directly trains the linear gate to recognize the essential Oracle ARF-win regime.
    """
    def __init__(
        self,
        tau_gate=0.25,
        max_rho=0.15,
        h_cooldown=100,
        buffer_b=200,
        recent_w=50,
        adwin_delta=0.005,
        adwin_clock=32,
        use_pruned_arf=True,
        seed=42
    ):
        self.name = "CGT-v3 (Learned Meta-Gate)"
        self.tau_gate = tau_gate
        self.max_rho = max_rho
        self.h_cooldown = h_cooldown
        self.buffer_b = buffer_b
        self.recent_w = recent_w
        self.use_pruned_arf = use_pruned_arf
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Base learners
        self.ht = HoeffdingTreeWrapper()
        self.arf = ARFOptimizedWrapper(seed=seed) if use_pruned_arf else ARFFullWrapper(seed=seed)
        self.detector = ADWIN(delta=adwin_delta, clock=adwin_clock)
        
        # Online meta-classifier: standardized logistic regression
        self.meta_gate = StandardScaler() | LogisticRegression()
        
        # State buffers
        self.buffer = deque(maxlen=buffer_b)
        self.recent_ht_errors = deque(maxlen=recent_w)
        self.cooldown_counter = 0
        self.gate_open = False
        
        # Diagnostic tracking
        self.total_steps = 0
        self.gate_open_steps = 0
        self.total_alarms = 0
        self.learned_gate_triggers = 0
        self.budget_throttled_steps = 0
        
        # Cache current instance features and HT proba for the step
        self._current_meta_features = None

    def _extract_meta_features(self, x):
        proba = self.ht.model.predict_proba_one(x)
        p0 = proba.get(0, 0.5)
        p1 = proba.get(1, 0.5)
        margin = abs(p1 - p0)
        
        err_rate = sum(self.recent_ht_errors) / len(self.recent_ht_errors) if self.recent_ht_errors else 0.15
        
        h_sin = x.get('hour_sin', 0.0)
        h_cos = x.get('hour_cos', 0.0)
        ren_ratio = x.get('renewable_ratio_lag_1', 0.2)
        d_diff_24 = x.get('demand_diff_1_24', 0.0) / 5000.0  # approximate scaling
        
        return {
            'margin': float(margin),
            'recent_err': float(err_rate),
            'hour_sin': float(h_sin),
            'hour_cos': float(h_cos),
            'ren_ratio': float(ren_ratio),
            'd_diff_24': float(d_diff_24),
            'in_cooldown': 1.0 if self.cooldown_counter > 0 else 0.0
        }

    def predict_one(self, x):
        self.total_steps += 1
        current_rho = self.gate_open_steps / max(self.total_steps, 1)
        budget_ok = current_rho < self.max_rho
        
        meta_feat = self._extract_meta_features(x)
        self._current_meta_features = meta_feat
        
        # Meta-classifier probability that ARF beats HT
        proba_meta = self.meta_gate.predict_proba_one(meta_feat)
        prob_arf_wins = proba_meta.get(1, 0.0)
        
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
            return self.arf.predict_one(x)
        return self.ht.predict_one(x)

    def learn_one(self, x, y, y_pred):
        error = 1.0 if y_pred != y else 0.0
        
        # 1. Update ADWIN detector
        self.detector.update(error)
        if self.detector.drift_detected:
            self.total_alarms += 1
            self.cooldown_counter = self.h_cooldown
            self.gate_open = True
            for bx, by in self.buffer:
                self.arf.learn_one(bx, by)
                
        # 2. Shadow evaluation for meta-training
        # We need true HT error and ARF error to create target z_t
        pred_ht = self.ht.predict_one(x)
        err_ht = 1.0 if pred_ht != y else 0.0
        self.recent_ht_errors.append(err_ht)
        
        # In online streaming, obtain ARF prediction
        pred_arf = self.arf.predict_one(x)
        err_arf = 1.0 if pred_arf != y else 0.0
        
        # Target: ARF is correct AND HT is wrong (the 8% Oracle advantage zone)
        z_t = 1 if (err_ht == 1.0 and err_arf == 0.0) else 0
        
        # Train meta-gate
        if self._current_meta_features is not None:
            self.meta_gate.learn_one(self._current_meta_features, z_t)
            
        # 3. Update base learners
        self.ht.learn_one(x, y)
        if self.gate_open:
            self.arf.learn_one(x, y)
        else:
            self.buffer.append((x, y))
            # 5% background probing update to keep ARF informed
            if self.rng.rand() < 0.05:
                self.arf.learn_one(x, y)

