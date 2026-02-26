"""
Online conformal prediction with adaptive quantile tracking.

In streaming/non-stationary settings, exchangeability is violated.
This module implements the Adaptive Conformal Inference (ACI) framework
of Gibbs & Candès (2021), which maintains long-run coverage by
dynamically adjusting the quantile level using online gradient descent:

    α_{t+1} = α_t + γ(α - err_t)

where err_t = 1{Y_t ∉ C_t(X_t)} and γ is a learning rate. Under mild
conditions, this achieves:

    (1/T) Σ_t err_t → α  as T → ∞

even under arbitrary distribution shift.

References
----------
[1] Gibbs, Candès (2021). "Adaptive Conformal Inference Under
    Distribution Shift." NeurIPS.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class OnlineState:
    """Snapshot of online conformal tracker state."""
    t: int
    alpha_t: float
    cumulative_coverage: float
    recent_coverage: float  # Last 100 steps
    target: float


class OnlineConformalTracker:
    """Adaptive conformal inference for streaming data.

    Maintains valid long-run coverage under distribution shift
    by adjusting the miscoverage level online.

    Parameters
    ----------
    alpha : float
        Target long-run miscoverage rate.
    gamma : float
        Learning rate for quantile adaptation. Larger = faster adaptation
        but more variance. Theory suggests γ ∈ [0.001, 0.05].
    window : int
        Number of recent scores to maintain for quantile estimation.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.005,
                 window: int = 500) -> None:
        self.alpha_target = alpha
        self.gamma = gamma
        self.window = window

        self._alpha_t = alpha
        self._scores: List[float] = []
        self._errors: List[int] = []
        self._t = 0
        self._alpha_history: List[float] = [alpha]

    def update(self, score: float, true_value: float,
               prediction: float) -> Tuple[float, bool]:
        """Process one observation and update the adaptive quantile.

        Parameters
        ----------
        score : float
            Nonconformity score for this observation.
        true_value : float
            True label/value.
        prediction : float
            Model prediction.

        Returns
        -------
        q_t : float
            Current quantile threshold.
        covered : bool
            Whether the true value was covered.
        """
        self._t += 1

        # Maintain rolling window of scores
        self._scores.append(score)
        if len(self._scores) > self.window:
            self._scores.pop(0)

        # Current quantile from recent scores
        if len(self._scores) >= 10:
            level = min(1 - self._alpha_t, 1.0)
            level = max(level, 0.0)
            q_t = float(np.quantile(self._scores, max(level, 0.01)))
        else:
            q_t = float('inf')  # Cover everything until enough data

        # Check coverage
        covered = score <= q_t
        err_t = 0 if covered else 1
        self._errors.append(err_t)

        # Online gradient descent on alpha
        # α_{t+1} = α_t + γ(α_target - err_t)
        self._alpha_t = self._alpha_t + self.gamma * (self.alpha_target - err_t)
        # Clip to valid range
        self._alpha_t = np.clip(self._alpha_t, 0.001, 0.999)
        self._alpha_history.append(self._alpha_t)

        return q_t, covered

    @property
    def state(self) -> OnlineState:
        """Current tracker state."""
        n = len(self._errors)
        recent = self._errors[-100:] if n >= 100 else self._errors
        return OnlineState(
            t=self._t,
            alpha_t=self._alpha_t,
            cumulative_coverage=1 - np.mean(self._errors) if n > 0 else 1.0,
            recent_coverage=1 - np.mean(recent) if recent else 1.0,
            target=1 - self.alpha_target,
        )

    @property
    def alpha_trajectory(self) -> np.ndarray:
        """Full history of adapted alpha values."""
        return np.array(self._alpha_history)

    def coverage_trajectory(self, window: int = 100) -> np.ndarray:
        """Rolling coverage rate over time."""
        errors = np.array(self._errors, dtype=np.float64)
        if len(errors) < window:
            return 1 - np.cumsum(errors) / np.arange(1, len(errors) + 1)
        kernel = np.ones(window) / window
        return 1 - np.convolve(errors, kernel, mode='valid')
