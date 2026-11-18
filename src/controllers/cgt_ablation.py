from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper

class CGTFixedScheduleAblation:
    """
    CGT Ablation (Slide 18): Fixed periodic activation schedule instead of ADWIN drift-triggering.
    Tests whether detector-driven gating is truly beneficial over static periodic switching.
    """
    def __init__(self, period=1000, active_duration=200, seed=42):
        self.name = "CGT Ablation (Fixed Schedule)"
        self.period = period
        self.active_duration = active_duration
        
        self.ht = HoeffdingTreeWrapper()
        self.arf = ARFFullWrapper(seed=seed)
        
        self.step = 0
        self.gate_open = False

    def predict_one(self, x):
        if self.gate_open:
            return self.arf.predict_one(x)
        return self.ht.predict_one(x)

    def learn_one(self, x, y, y_pred):
        self.step += 1
        # Gate open periodically for active_duration every period
        cycle_pos = self.step % self.period
        self.gate_open = (cycle_pos < self.active_duration)
        
        # Training rule: HT always learns, ARF learns only when gate is open
        self.ht.learn_one(x, y)
        if self.gate_open:
            self.arf.learn_one(x, y)
