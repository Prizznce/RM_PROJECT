from collections import deque
import time
from river.drift import ADWIN
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper, ARFOptimizedWrapper

class ConditionalGatedTrainingController:
    """
    Proposed Method: Conditional Gated Training (CGT).
    ADWIN acts as an interrupt; ARF stays dormant until concept drift is detected.
    
    Gate Rule:
      g_t = 1 if alarm_t or (g_{t-1} == 1 and not Stable_t) else 0
      y_hat_t = ARF(x_t) if g_t == 1 else HT(x_t)
      
    Warm-Start Strategies:
      S1: Resume ARF's frozen state and replay the last B buffered instances.
      S2: Reset ARF and train on the last B buffered instances.
      S3: Resume frozen state with no replay.
      
    Parameters:
      use_pruned_arf: If True, uses 5-tree Optimized ARF for 2x faster wake-up and low latency.
    """
    def __init__(self, stability_k=500, buffer_b=200, warm_start_strategy="S1",
                 adwin_delta=0.002, adwin_clock=32, use_pruned_arf=False, seed=42):
        self.use_pruned_arf = use_pruned_arf
        prefix = "CGT-Plus (Optimized)" if use_pruned_arf else "CGT (Proposed)"
        self.name = f"{prefix}, {warm_start_strategy}" if not use_pruned_arf else "CGT-Plus (Optimized Hybrid)"
        self.stability_k = stability_k
        self.buffer_b = buffer_b
        self.warm_start_strategy = warm_start_strategy
        self.seed = seed
        
        self.ht = HoeffdingTreeWrapper()
        self.arf = ARFOptimizedWrapper(seed=seed) if use_pruned_arf else ARFFullWrapper(seed=seed)
        self.detector = ADWIN(delta=adwin_delta, clock=adwin_clock)
        
        self.gate_open = False
        self.stable_counter = 0
        self.buffer = deque(maxlen=buffer_b)
        
        # Diagnostic counters
        self.total_alarms = 0
        self.gate_open_steps = 0
        self.total_steps = 0
        self.replay_instances_trained = 0

    def predict_one(self, x):
        """Active learner makes prediction before label arrives."""
        if self.gate_open:
            return self.arf.predict_one(x)
        return self.ht.predict_one(x)

    def learn_one(self, x, y, y_pred):
        """
        Prequential evaluation step:
        Update ADWIN with 0/1 prediction error, update gate state, train models.
        """
        self.total_steps += 1
        error = 1.0 if y_pred != y else 0.0
        
        # 1. Update ADWIN detector
        self.detector.update(error)
        alarm = self.detector.drift_detected
        
        if alarm:
            self.total_alarms += 1
            self.stable_counter = 0  # reset stability horizon
            
            # If gate was closed, wake up ARF with warm-start strategy
            if not self.gate_open:
                self.gate_open = True
                self._execute_warm_start()
        else:
            if self.gate_open:
                self.stable_counter += 1
                # If stable for K consecutive instances, re-lock ARF to dormant
                if self.stable_counter >= self.stability_k:
                    self.gate_open = False
                    self.stable_counter = 0

        # 2. Training Rule:
        # HT is trained on EVERY instance
        self.ht.learn_one(x, y)
        
        # ARF is trained ONLY while gate is open
        if self.gate_open:
            self.gate_open_steps += 1
            self.arf.learn_one(x, y)
        else:
            # While ARF is dormant, maintain FIFO buffer for future warm-starts
            self.buffer.append((x, y))

    def _execute_warm_start(self):
        """Executes S1, S2, or S3 warm-start upon gate opening."""
        if self.warm_start_strategy == "S1":
            # Resume frozen state and replay last B buffered instances
            for buf_x, buf_y in self.buffer:
                self.arf.learn_one(buf_x, buf_y)
                self.replay_instances_trained += 1
        elif self.warm_start_strategy == "S2":
            # Reset ARF and train from scratch on buffer
            self.arf = ARFOptimizedWrapper(seed=self.seed) if self.use_pruned_arf else ARFFullWrapper(seed=self.seed)
            for buf_x, buf_y in self.buffer:
                self.arf.learn_one(buf_x, buf_y)
                self.replay_instances_trained += 1
        elif self.warm_start_strategy == "S3":
            # Resume frozen state with zero replay
            pass
        else:
            raise ValueError(f"Unknown warm start strategy: {self.warm_start_strategy}")
