from collections import deque
from river.linear_model import LogisticRegression
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper

class A4LearnedRouter:
    """
    A4: Learned Router (Slide 9).
    Learns an online linear classifier f_theta(x_b) mapping batch features
    x_b = [E_b, V_b, D_b, C_b, Delta_E_b] to arm with minimal loss J(m) = E(m) + lambda_C * C(m).
    """
    def __init__(self, batch_size=50, lambda_c=0.3, seed=42):
        self.name = "A4: Learned Router"
        self.batch_size = batch_size
        self.lambda_c = lambda_c
        
        self.ht = HoeffdingTreeWrapper()
        self.arf = ARFFullWrapper(seed=seed)
        
        # Meta-classifier
        self.router = LogisticRegression()
        
        self.active_arm = "HT"
        self.batch_count = 0
        self.batch_errors_ht = []
        self.batch_errors_arf = []
        self.prev_error_ht = 0.5
        self.costs = {"HT": 0.1, "ARF": 1.0}

    def predict_one(self, x):
        if self.active_arm == "ARF":
            return self.arf.predict_one(x)
        return self.ht.predict_one(x)

    def learn_one(self, x, y, y_pred):
        # Shadow evaluate both arms periodically for router supervision
        pred_ht = self.ht.predict_one(x)
        err_ht = 1.0 if pred_ht != y else 0.0
        self.batch_errors_ht.append(err_ht)
        
        # Train active arm & HT
        self.ht.learn_one(x, y)
        if self.active_arm == "ARF":
            self.arf.learn_one(x, y)
            err_arf = 1.0 if y_pred != y else 0.0
            self.batch_errors_arf.append(err_arf)
        else:
            # During HT phases, estimate ARF error from recent window
            self.batch_errors_arf.append(err_ht * 0.8) # prior estimate
            
        self.batch_count += 1
        if self.batch_count >= self.batch_size:
            # Compute batch summary features
            e_ht = sum(self.batch_errors_ht) / len(self.batch_errors_ht)
            e_arf = sum(self.batch_errors_arf) / len(self.batch_errors_arf)
            
            delta_e = e_ht - self.prev_error_ht
            self.prev_error_ht = e_ht
            
            # Loss J(m) = E(m) + lambda_C * C(m)
            j_ht = e_ht + self.lambda_c * self.costs["HT"]
            j_arf = e_arf + self.lambda_c * self.costs["ARF"]
            
            # Target arm: 1 for ARF, 0 for HT
            optimal_arm_idx = 1 if j_arf < j_ht else 0
            
            meta_features = {
                'e_b': float(e_ht),
                'v_b': float(abs(delta_e)),
                'delta_e': float(delta_e),
                'cost_ratio': float(self.costs['HT'] / self.costs['ARF'])
            }
            
            # Train router
            self.router.learn_one(meta_features, optimal_arm_idx)
            
            # Predict for next batch
            router_pred = self.router.predict_one(meta_features)
            self.active_arm = "ARF" if router_pred == 1 else "HT"
            
            # Reset batch buffers
            self.batch_count = 0
            self.batch_errors_ht.clear()
            self.batch_errors_arf.clear()
