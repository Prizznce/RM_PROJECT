"""
Drift-Adaptive Conformal Gate (online conformal risk control on the exit threshold).

Gate rule (margin-based early exit):
    g_t = 1  iff  m_t < tau_t   (or an ADWIN drift flag is active - handled by the controller)

Threshold update after y_t is revealed (Gibbs & Candes-style adaptive step):
    tau_{t+1} = clip(tau_t + gamma * (l_t - alpha), 0, 1)

    l_t > alpha  ->  tau grows   ->  more samples expand to trees 4-10
    l_t < alpha  ->  tau shrinks ->  compute is saved

Loss definitions (``loss_mode``)
--------------------------------
"final_error"  l_t = cost(y_hat_t, y_t)                (the literal spec)
               Long-run behaviour: mean(l) -> alpha only if the full 10-tree
               ensemble can reach risk alpha. If its error rate is above alpha,
               tau saturates at 1 and the gate degenerates to Always-ARF-10.

"exit_error"   l_t = (1 - g_t) * cost(y_hat_core_t, y_t)
               = risk of an error that *exited early without escalation*.
               Always achievable (g = 1 everywhere gives l = 0), so the online
               update guarantees  (1/T) sum l_t -> alpha  (up to O(1/(gamma T)))
               for every alpha >= 0, i.e. alpha directly budgets the fraction
               of un-escalated mistakes.

Asymmetric cost (``w_fn``): a missed positive (y=1, y_hat=0) costs w_fn, a
false alarm (y=0, y_hat=1) costs 1, normalised by max(w_fn, 1) into [0, 1].
w_fn = 1 recovers the standard 0-1 error.
"""

from __future__ import annotations

LOSS_MODES = ("final_error", "exit_error")


def asymmetric_cost(y_hat: int, y: int, w_fn: float) -> float:
    """Normalised asymmetric 0-1 cost in [0, 1]."""
    if y_hat == y:
        return 0.0
    norm = max(float(w_fn), 1.0)
    if y == 1 and y_hat == 0:  # false negative (missed stress event)
        return float(w_fn) / norm
    return 1.0 / norm  # false positive


class DriftAdaptiveConformalGate:
    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 0.01,
        tau_init: float = 0.30,
        w_fn: float = 1.0,
        loss_mode: str = "exit_error",
    ):
        if loss_mode not in LOSS_MODES:
            raise ValueError(f"loss_mode must be one of {LOSS_MODES}, got {loss_mode!r}")
        if not 0.0 <= tau_init <= 1.0:
            raise ValueError("tau_init must lie in [0, 1]")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.tau = float(tau_init)
        self.w_fn = float(w_fn)
        self.loss_mode = loss_mode

        self.n_updates = 0
        self.cum_loss = 0.0

    def is_uncertain(self, margin: float) -> bool:
        return margin < self.tau

    def step_loss(self, gated: bool, y_hat_final: int, y_hat_core: int, y: int) -> float:
        if self.loss_mode == "final_error":
            return asymmetric_cost(y_hat_final, y, self.w_fn)
        # exit_error
        return 0.0 if gated else asymmetric_cost(y_hat_core, y, self.w_fn)

    def update(self, gated: bool, y_hat_final: int, y_hat_core: int, y: int) -> float:
        loss = self.step_loss(gated, y_hat_final, y_hat_core, y)
        self.tau = min(1.0, max(0.0, self.tau + self.gamma * (loss - self.alpha)))
        self.n_updates += 1
        self.cum_loss += loss
        return loss

    @property
    def mean_loss(self) -> float:
        return self.cum_loss / max(self.n_updates, 1)


class StaticMarginGate:
    """Ablation: fixed threshold, no adaptation (same interface)."""

    def __init__(self, tau: float = 0.30, w_fn: float = 1.0):
        self.tau = float(tau)
        self.w_fn = float(w_fn)
        self.alpha = None
        self.gamma = 0.0
        self.loss_mode = "static"
        self.n_updates = 0
        self.cum_loss = 0.0

    def is_uncertain(self, margin: float) -> bool:
        return margin < self.tau

    def update(self, gated: bool, y_hat_final: int, y_hat_core: int, y: int) -> float:
        loss = asymmetric_cost(y_hat_final, y, self.w_fn)
        self.n_updates += 1
        self.cum_loss += loss
        return loss

    @property
    def mean_loss(self) -> float:
        return self.cum_loss / max(self.n_updates, 1)
