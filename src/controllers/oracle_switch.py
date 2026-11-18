import time
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper

class OracleGatedSwitch:
    """
    Theoretical Oracle Switch Upper Bound (Slide 9 & Day 2 Diagnostic).
    
    An omniscient meta-controller that observes the true label y_t at step t:
    - If HT is correct: Select HT (Cost = HT cost, Error = 0).
    - If HT is wrong and ARF is correct: Select ARF (Cost = ARF cost, Error = 0).
    - If both are wrong: Select HT (Cost = HT cost, Error = 1, avoids wasting ARF compute).
    
    This establishes the absolute theoretical ceiling of any gating scheme
    between HT and ARF on the data stream, providing:
    1. Maximum possible achievable accuracy.
    2. Minimum necessary ARF activation fraction (rho_oracle).
    3. The Pareto upper frontier.
    """
    def __init__(self, seed=42):
        self.name = "Oracle Switch (Upper Bound)"
        self.ht = HoeffdingTreeWrapper()
        self.arf = ARFFullWrapper(seed=seed)
        
        self.total_steps = 0
        self.gate_open_steps = 0
        self.gate_open = False
        self.active_arm = "HT"
        
        # Diagnostic tracking
        self.both_correct_count = 0
        self.arf_only_count = 0
        self.ht_only_count = 0
        self.neither_count = 0

    def predict_one(self, x):
        """
        In standard prequential test-then-train, the true label is revealed during learn_one.
        For prequential logging of predict_one, Oracle peeks or buffers the prediction.
        To strictly satisfy the prequential interface, we compute both candidate predictions.
        Note: The actual selection is finalized when y_t is observed in learn_one.
        """
        # Return HT by default for the pre-label query; the effective oracle prediction
        # is logged in learn_one.
        return self.ht.predict_one(x)

    def evaluate_and_learn(self, x, y):
        """
        Omniscient step: evaluates both arms against y, routes to optimal arm,
        and trains base learners.
        
        Returns:
            y_oracle: Prediction selected by Oracle
            chosen_arm: 'HT' or 'ARF'
            cost_factor: Latency multiplier (1 for HT, ~14 for ARF)
        """
        self.total_steps += 1
        
        p_ht = self.ht.predict_one(x)
        p_arf = self.arf.predict_one(x)
        
        ht_ok = (p_ht == y)
        arf_ok = (p_arf == y)
        
        if ht_ok:
            # HT is correct: always pick HT (cheapest)
            chosen_arm = "HT"
            y_oracle = p_ht
            self.gate_open = False
            if arf_ok:
                self.both_correct_count += 1
            else:
                self.ht_only_count += 1
        elif arf_ok:
            # HT is wrong, but ARF is correct: switch to ARF (essential win)
            chosen_arm = "ARF"
            y_oracle = p_arf
            self.gate_open = True
            self.gate_open_steps += 1
            self.arf_only_count += 1
        else:
            # Both wrong: pick HT to minimize computational loss
            chosen_arm = "HT"
            y_oracle = p_ht
            self.gate_open = False
            self.neither_count += 1
            
        self.active_arm = chosen_arm
        
        # Always keep both models updated so Oracle observes their optimal current state
        self.ht.learn_one(x, y)
        self.arf.learn_one(x, y)
        
        return y_oracle, chosen_arm

    def learn_one(self, x, y, y_pred=None):
        # Learn from stream instance
        self.ht.learn_one(x, y)
        self.arf.learn_one(x, y)

