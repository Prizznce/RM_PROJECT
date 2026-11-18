import numpy as np
from river import linear_model, neighbors, tree, ensemble, preprocessing, optim

class NaivePersistenceHourly:
    """
    Mandatory Baseline: 1-hour lag persistence (Directional Momentum).
    Predicts that current hour's direction equals previous hour's direction:
    y_hat_t = 1 if Demand_{t-1} > Demand_{t-2} else 0.
    """
    def __init__(self):
        self.name = "Persistence (1-Hr Momentum)"

    def predict_one(self, x):
        # x contains demand_diff_1_2 = Demand_{t-1} - Demand_{t-2}
        return 1 if x.get('demand_diff_1_2', 0.0) > 0 else 0

    def learn_one(self, x, y):
        pass  # Stateless baseline


class NaivePersistenceDiurnal:
    """
    Mandatory Baseline: 24-hour diurnal cycle persistence.
    Predicts that current demand direction matches the 24-hr diurnal net movement:
    y_hat_t = 1 if Demand_{t-1} > Demand_{t-24} else 0.
    """
    def __init__(self):
        self.name = "Persistence (24-Hr Diurnal)"

    def predict_one(self, x):
        # x contains demand_diff_1_24 = Demand_{t-1} - Demand_{t-24}
        return 1 if x.get('demand_diff_1_24', 0.0) > 0 else 0

    def learn_one(self, x, y):
        pass  # Stateless baseline


class ZeroSkillMajorityClass:
    """Zero-skill majority class predictor (always predicts 0)."""
    def __init__(self):
        self.name = "Zero-Skill Majority (Always 0)"

    def predict_one(self, x):
        return 0

    def learn_one(self, x, y):
        pass


class StreamingSGDClassifier:
    """
    Online Logistic Regression with Stochastic Gradient Descent (SGD).
    Includes online z-score feature normalization (StandardScaler).
    """
    def __init__(self, lr=0.01):
        self.name = "Online SGD (Logistic Regression)"
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
            optimizer=optim.SGD(lr=lr)
        )

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y):
        self.model.learn_one(x, y)


class AdaptiveHoeffdingTreeWrapper:
    """
    Adaptive Hoeffding Tree (HAT / HATT, Bifet & Gavalda, 2009).
    Uses built-in ADWIN drift estimators at every split node to prune/replace subtrees.
    """
    def __init__(self, grace_period=200, delta=1e-7, seed=42):
        self.name = "Adaptive Hoeffding Tree (HAT)"
        self.model = tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=grace_period,
            delta=delta,
            seed=seed
        )

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y):
        self.model.learn_one(x, y)


class StreamingKNNWrapper:
    """
    Streaming k-Nearest Neighbors with Sliding Window (LazySearch, window=100).
    Features normalized with online StandardScaler.
    """
    def __init__(self, n_neighbors=5, window_size=100):
        self.name = f"Streaming kNN (k={n_neighbors}, W={window_size})"
        self.model = preprocessing.StandardScaler() | neighbors.KNNClassifier(
            n_neighbors=n_neighbors,
            engine=neighbors.LazySearch(window_size=window_size)
        )

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y):
        self.model.learn_one(x, y)


class LeveragingBaggingWrapper:
    """
    Leveraging Bagging (Bifet et al., 2010).
    Ensemble of Hoeffding Trees with Poisson(lambda) resample weights.
    """
    def __init__(self, n_models=5, grace_period=100, seed=42):
        self.name = f"Leveraging Bagging ({n_models} trees)"
        self.model = ensemble.LeveragingBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(grace_period=grace_period),
            n_models=n_models,
            seed=seed
        )

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y):
        self.model.learn_one(x, y)


class StreamingRandomPatchesWrapper:
    """
    Streaming Random Patches (SRP, Gomes et al., 2019).
    Ensemble with subspace feature randomization and adaptive drift detection.
    """
    def __init__(self, n_models=5, grace_period=100, seed=42):
        self.name = f"Streaming Random Patches (SRP, {n_models} trees)"
        self.model = ensemble.SRPClassifier(
            model=tree.HoeffdingTreeClassifier(grace_period=grace_period),
            n_models=n_models,
            seed=seed
        )

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y):
        self.model.learn_one(x, y)

