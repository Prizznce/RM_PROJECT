import math
from collections import deque
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper

class A2UCBController:
    """
    A2: Cost-aware Score with UCB Selection (Slide 7, OMS-MAB with compute cost).
    Scores each learner: S_m = lambda_A * A_m - lambda_V * V_m + lambda_D * D_m - lambda_C * C_m
    UCB_m = S_bar_m + c * sqrt(ln(t) / N_m)
    """
    def __init__(self, c=0.5, lambda_acc=0.3, lambda_v=0.2, lambda_d=0.3, lambda_c=0.2,
                 discount=0.98, window_size=50, seed=42):
        self.name = "A2: Cost-aware UCB"
        self.c = c
        self.lambda_acc = lambda_acc
        self.lambda_v = lambda_v
        self.lambda_d = lambda_d
        self.lambda_c = lambda_c
        self.discount = discount
        self.window_size = window_size
        
        self.ht = HoeffdingTreeWrapper()
        self.arf = ARFFullWrapper(seed=seed)
        self.arms = ["HT", "ARF"]
        
        # Relative normalized cost: HT is approx 0.1, ARF is 1.0
        self.norm_costs = {"HT": 0.1, "ARF": 1.0}
        
        self.t = 0
        self.pulls = {"HT": 1.0, "ARF": 1.0}
        self.rewards = {"HT": 0.5, "ARF": 0.5}
        self.history = {"HT": deque(maxlen=window_size), "ARF": deque(maxlen=window_size)}
        
        self.selected_arm = "HT"

    def _compute_score(self, arm):
        hist = self.history[arm]
        if not hist:
            return 0.5
        acc = sum(hist) / len(hist)
        # Instability: variance of accuracy
        v = (sum((x - acc)**2 for x in hist) / len(hist)) if len(hist) > 1 else 0.0
        # Drift resilience: 1 - recent degradation
        d = 1.0 - v
        c = self.norm_costs[arm]
        return self.lambda_acc * acc - self.lambda_v * v + self.lambda_d * d - self.lambda_c * c

    def predict_one(self, x):
        self.t += 1
        # UCB selection
        best_arm = None
        best_ucb = -float("inf")
        ln_t = math.log(max(self.t, 2))
        
        for arm in self.arms:
            s_bar = self.rewards[arm]
            n_m = self.pulls[arm]
            ucb = s_bar + self.c * math.sqrt(ln_t / n_m)
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
                
        self.selected_arm = best_arm
        if self.selected_arm == "ARF":
            return self.arf.predict_one(x)
        return self.ht.predict_one(x)

    def learn_one(self, x, y, y_pred):
        is_correct = 1.0 if y_pred == y else 0.0
        self.history[self.selected_arm].append(is_correct)
        
        # Train active arm (and HT background)
        self.ht.learn_one(x, y)
        if self.selected_arm == "ARF":
            self.arf.learn_one(x, y)
            
        # Update discounted reward and pull count
        instant_score = self._compute_score(self.selected_arm)
        # Apply discount to decay old concepts
        for arm in self.arms:
            self.pulls[arm] *= self.discount
            self.rewards[arm] *= self.discount
            
        self.pulls[self.selected_arm] += 1.0
        # Running exponential update
        self.rewards[self.selected_arm] += (instant_score - self.rewards[self.selected_arm]) / self.pulls[self.selected_arm]
