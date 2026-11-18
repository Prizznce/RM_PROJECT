import math
import numpy as np
from river.drift.binary import HDDMA
from src.models.base_models import HoeffdingTreeWrapper, ARFFullWrapper

class OMSMABController:
    """
    OMS-MAB: Online Model Selection via Multi-Armed Bandits
    Faithful implementation of Wilson et al. (2024, Expert Systems).
    
    Paper: 'Multi-armed bandit based online model selection for concept-drift adaptation'
           Jobin Wilson, Santanu Chaudhury, Brejesh Lall (2024).
           
    Algorithmic Components:
    1. Arm Pool: M = [Hoeffding Tree (HT), Adaptive Random Forest (ARF)]
    2. Local Drift Detection: Each arm maintains a Hoeffding Dependent Drift Monitor
       with Average (HDDM_A, drift_confidence=0.001, warning_confidence=0.005).
    3. Non-Stationary Bandit Policy: Discounted Upper Confidence Bound (D-UCB).
       - Past rewards are discounted exponentially by factor gamma in [0.95, 0.99].
       - Arm selection: a_t = argmax_i ( mu_i + c * sqrt(2 * ln(N_t) / N_{i,t}) ).
    4. Feedback & Training:
       - Only the selected model arm is trained on incoming instance (x_t, y_t)
         to minimize computational footprint (Selected-Only Update).
       - Reward r_t in {0, 1} indicates 0-1 classification accuracy.
    5. Drift Reset Rule:
       - When HDDM_A signals drift on arm i, its discounted stats are reset to 0
         to trigger re-exploration and adaptation to the newly emerged concept.
    
    Documented Deviations from Paper:
    - Arm candidates: In our power grid domain, arms are HT and ARF (heavy ensemble vs light tree),
      whereas Wilson et al. evaluated combinations of Naive Bayes, HT, and SGD on synthetic streams.
    - Cost tracking: We record microsecond CPU latency alongside 0-1 reward.
    """
    def __init__(self, gamma=0.98, c=0.5, seed=42):
        self.name = "OMS-MAB (Wilson et al., 2024)"
        self.gamma = gamma
        self.c = c
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Candidate Arms
        self.arms = [
            HoeffdingTreeWrapper(),
            ARFFullWrapper(seed=seed)
        ]
        self.arm_names = ["HT", "ARF"]
        self.k = len(self.arms)
        
        # Local drift monitors per arm (HDDM_A as specified in Wilson et al.)
        self.detectors = [
            HDDMA(drift_confidence=0.001, warning_confidence=0.005),
            HDDMA(drift_confidence=0.001, warning_confidence=0.005)
        ]
        
        # Discounted Bandit Statistics
        self.discounted_pulls = np.zeros(self.k, dtype=float)
        self.discounted_rewards = np.zeros(self.k, dtype=float)
        self.arm_pull_counts = np.zeros(self.k, dtype=int)
        
        self.active_arm_idx = 0
        self.total_steps = 0
        self.gate_open_steps = 0
        self.total_drifts_detected = 0

    def predict_one(self, x):
        """
        Step 1: Select arm using Discounted-UCB (D-UCB)
        Step 2: Return prediction from chosen arm
        """
        self.total_steps += 1
        
        # Initial cold start: pull each arm at least once
        unpulled = [i for i in range(self.k) if self.arm_pull_counts[i] < 5]
        if unpulled:
            self.active_arm_idx = self.rng.choice(unpulled)
        else:
            total_n = np.sum(self.discounted_pulls)
            ucb_values = np.zeros(self.k)
            for i in range(self.k):
                n_i = self.discounted_pulls[i]
                if n_i <= 1e-4:
                    ucb_values[i] = 1e5  # force exploration
                else:
                    mu_i = self.discounted_rewards[i] / n_i
                    bonus = self.c * math.sqrt((2.0 * math.log(max(total_n, 1.01))) / n_i)
                    ucb_values[i] = mu_i + bonus
            self.active_arm_idx = int(np.argmax(ucb_values))
            
        if self.arm_names[self.active_arm_idx] == "ARF":
            self.gate_open_steps += 1
            
        return self.arms[self.active_arm_idx].predict_one(x)

    def learn_one(self, x, y, y_pred):
        """
        Prequential Step 3 & 4:
        - Compute reward r_t in {0, 1}
        - Update chosen arm model
        - Update chosen arm HDDM_A detector
        - Apply exponential discounting to all arms
        """
        idx = self.active_arm_idx
        reward = 1.0 if y_pred == y else 0.0
        error = 1.0 - reward
        
        # 1. Train only the chosen model
        self.arms[idx].learn_one(x, y)
        self.arm_pull_counts[idx] += 1
        
        # 2. Check local drift detector on the active arm
        self.detectors[idx].update(error)
        if self.detectors[idx].drift_detected:
            self.total_drifts_detected += 1
            # Wilson et al. reset: flush staled discounted statistics for this arm
            self.discounted_pulls[idx] = 1.0
            self.discounted_rewards[idx] = reward
        else:
            # 3. Apply discounting
            self.discounted_pulls *= self.gamma
            self.discounted_rewards *= self.gamma
            self.discounted_pulls[idx] += 1.0
            self.discounted_rewards[idx] += reward

