import time
from river.tree import HoeffdingTreeClassifier
from river.forest import ARFClassifier

class HoeffdingTreeWrapper:
    """Always-on lightweight base learner."""
    def __init__(self, grace_period=200, delta=1e-7):
        self.grace_period = grace_period
        self.delta = delta
        self.model = HoeffdingTreeClassifier(grace_period=grace_period, delta=delta)
        self.name = "Always-HT"

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y):
        self.model.learn_one(x, y)


class ARFFullWrapper:
    """Always-on heavy ensemble reference learner (ARF)."""
    def __init__(self, n_models=10, lambda_value=6, grace_period=50, delta=0.01, seed=42):
        self.n_models = n_models
        self.lambda_value = lambda_value
        self.grace_period = grace_period
        self.delta = delta
        self.seed = seed
        self.name = "Always-ARF"
        self.model = ARFClassifier(
            n_models=n_models,
            lambda_value=lambda_value,
            grace_period=grace_period,
            delta=delta,
            seed=seed
        )

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y):
        self.model.learn_one(x, y)


class ARFOptimizedWrapper:
    """A3: Optimized ARF with pruned ensemble size (5 trees) for reduced CPU footprint."""
    def __init__(self, n_models=5, lambda_value=6, grace_period=50, delta=0.01, seed=42):
        self.n_models = n_models
        self.lambda_value = lambda_value
        self.grace_period = grace_period
        self.delta = delta
        self.seed = seed
        self.name = "A3: Optimized ARF"
        self.model = ARFClassifier(
            n_models=n_models,
            lambda_value=lambda_value,
            grace_period=grace_period,
            delta=delta,
            seed=seed
        )

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return 0 if pred is None else int(pred)

    def learn_one(self, x, y):
        self.model.learn_one(x, y)
