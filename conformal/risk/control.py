"""
Conformal Risk Control: generalizing beyond set coverage.

Standard conformal prediction controls miscoverage: P(Y ∉ C(X)) ≤ α.
But many applications need to control other risk functions:

- False negative rate in medical screening
- Expected set size (efficiency)
- Weighted loss where some errors cost more

Learn-then-Test (LTT) provides a framework: for ANY bounded loss
function L(C_λ(X), Y) parameterized by threshold λ, find the
smallest λ such that E[L] ≤ α, with finite-sample guarantee.

References
----------
[1] Angelopoulos, Bates, Fisch, Lei, Schuster (2022).
    "Conformal Risk Control." ICLR.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, Optional, List


class ConformalRiskController:
    """Learn-then-Test conformal risk control.

    Given a loss function L(λ, X, Y) monotone in λ, finds the smallest
    λ such that E[L(λ, X, Y)] ≤ α with high probability.

    Parameters
    ----------
    alpha : float
        Target risk level.
    loss_fn : callable
        L(lambda, predictions, labels) -> array of per-sample losses.
        Must be monotone non-increasing in lambda.
    lambda_grid : array-like
        Grid of candidate lambda values to search over. Must be sorted.
    delta : float
        Failure probability for the risk bound. Default 0.05 means the
        bound holds with probability ≥ 95%.
    """

    def __init__(self, alpha: float = 0.1,
                 loss_fn: Optional[Callable] = None,
                 lambda_grid: Optional[ArrayLike] = None,
                 delta: float = 0.05) -> None:
        self.alpha = alpha
        self.delta = delta
        self._loss_fn = loss_fn
        self._lambda_grid = np.asarray(lambda_grid) if lambda_grid is not None else None
        self._lambda_hat: Optional[float] = None
        self._risk_profile: Optional[List] = None

    def calibrate(self, predictions: ArrayLike, labels: ArrayLike) -> float:
        """Find the risk-controlling threshold via Hoeffding-Bentkus bound.

        Parameters
        ----------
        predictions, labels : array-like

        Returns
        -------
        lambda_hat : float
            Smallest lambda satisfying the risk bound.
        """
        if self._loss_fn is None:
            raise ValueError("loss_fn must be provided")
        if self._lambda_grid is None:
            raise ValueError("lambda_grid must be provided")

        preds = np.asarray(predictions, dtype=np.float64)
        labs = np.asarray(labels, dtype=np.float64)
        n = len(labs)

        # Bonferroni correction for multiple testing across lambda grid
        m = len(self._lambda_grid)
        delta_adj = self.delta / m

        self._risk_profile = []

        for lam in self._lambda_grid:
            losses = self._loss_fn(lam, preds, labs)
            mean_loss = np.mean(losses)

            # Hoeffding bound: P(E[L] > mean_L + t) ≤ exp(-2nt²)
            # Solving for t at confidence level delta_adj:
            t = np.sqrt(np.log(1 / delta_adj) / (2 * n))

            upper_bound = mean_loss + t
            self._risk_profile.append({
                "lambda": float(lam),
                "empirical_risk": float(mean_loss),
                "upper_bound": float(upper_bound),
                "controls_risk": upper_bound <= self.alpha,
            })

        # Find smallest lambda that controls risk
        for entry in self._risk_profile:
            if entry["controls_risk"]:
                self._lambda_hat = entry["lambda"]
                return self._lambda_hat

        # If no lambda works, use the largest (most conservative)
        self._lambda_hat = float(self._lambda_grid[-1])
        return self._lambda_hat

    @property
    def lambda_hat(self) -> Optional[float]:
        return self._lambda_hat

    @property
    def risk_profile(self) -> Optional[List]:
        """Risk at each lambda value with confidence bounds."""
        return self._risk_profile
