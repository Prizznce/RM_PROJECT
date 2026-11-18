from collections import deque
import time
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper

class A1ThresholdController:
    """
    A1: Threshold-based switching with hysteresis (Slide 6).
    m_t = ARF if E_t >= theta_hi
          HT  if E_t <= theta_lo
          m_{t-1} otherwise
    """
    def __init__(self, theta_hi=0.25, theta_lo=0.15, window_size=50, seed=42):
        self.name = "A1: Threshold Switch"
        self.theta_hi = theta_hi
        self.theta_lo = theta_lo
        self.window_size = window_size
        
        self.ht = HoeffdingTreeWrapper()
        self.arf = ARFFullWrapper(seed=seed)
        
        self.active_model = "HT"
        self.error_window = deque(maxlen=window_size)
        
    def predict_one(self, x):
        if self.active_model == "ARF":
            return self.arf.predict_one(x)
        return self.ht.predict_one(x)

    def learn_one(self, x, y, y_pred):
        error = 1 if y_pred != y else 0
        self.error_window.append(error)
        
        # Train active model (and keep HT learning as lightweight backbone)
        self.ht.learn_one(x, y)
        if self.active_model == "ARF":
            self.arf.learn_one(x, y)
            
        # Hysteresis switching policy
        if len(self.error_window) >= 10:
            e_t = sum(self.error_window) / len(self.error_window)
            if e_t >= self.theta_hi:
                self.active_model = "ARF"
            elif e_t <= self.theta_lo:
                self.active_model = "HT"
