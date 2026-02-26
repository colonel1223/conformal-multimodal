"""
Regularized Adaptive Prediction Sets (RAPS).

For classification with K classes and softmax outputs π(x) ∈ Δ^K,
RAPS produces prediction sets that:
- Include the true class with probability ≥ 1-α
- Are small for easy inputs (peaked softmax) and larger for ambiguous ones
- Are regularized to avoid pathologically large sets

The score function is:

    s(x, y) = Σ_{j=1}^{o(y)} π_{(j)}(x) + λ·max(0, o(y) - k_reg) + u·π_{(o(y))}(x)

where o(y) is the rank of class y in sorted softmax, π_{(j)} is the j-th
largest softmax value, and u ~ Uniform(0,1) provides randomization for
exact (not just conservative) coverage.

References
----------
[1] Angelopoulos, Bates, Malik, Jordan. "Uncertainty Sets for Image
    Classifiers using Conformal Prediction." ICLR, 2021.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, List
from ..exceptions import NotCalibratedError, InsufficientDataError


class AdaptivePredictionSets:
    """RAPS: prediction sets with coverage guarantee and size regularization.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    k_reg : int
        Regularization kicks in after k_reg classes. Controls set size.
    lambda_reg : float
        Regularization strength. Higher → smaller sets, risk of undercoverage
        if set too high relative to the calibration set.
    randomize : bool
        If True, use randomized scores for exact (not just conservative) coverage.
    """

    def __init__(self, alpha: float = 0.1, k_reg: int = 5,
                 lambda_reg: float = 0.01, randomize: bool = True) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
        self.randomize = randomize
        self._q_hat: Optional[float] = None
        self._rng = np.random.default_rng()

    def _score_single(self, softmax: np.ndarray, label: int) -> float:
        """RAPS score for one example."""
        sorted_idx = np.argsort(-softmax)
        sorted_probs = softmax[sorted_idx]
        cumsum = np.cumsum(sorted_probs)

        rank = int(np.where(sorted_idx == label)[0][0])

        # Randomized tie-breaking for exact coverage
        u = self._rng.uniform() if self.randomize else 1.0

        # Cumulative probability up to true label (with randomization)
        score = cumsum[rank] - sorted_probs[rank] * (1 - u)

        # Regularization penalty
        score += self.lambda_reg * max(0, rank + 1 - self.k_reg)

        return float(score)

    def calibrate(self, softmax_outputs: ArrayLike, labels: ArrayLike) -> float:
        """Calibrate on held-out data.

        Parameters
        ----------
        softmax_outputs : array-like, shape (n, n_classes)
            Softmax probability vectors.
        labels : array-like, shape (n,)
            Integer class indices.
        """
        softmax = np.asarray(softmax_outputs, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.int64)
        n = len(labels_arr)

        min_n = int(np.ceil(1.0 / self.alpha)) - 1
        if n < min_n:
            raise InsufficientDataError(n, min_n)

        scores = np.array([self._score_single(softmax[i], labels_arr[i])
                           for i in range(n)])

        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self._q_hat = float(np.quantile(scores, level, method="higher"))
        return self._q_hat

    def predict(self, softmax_outputs: ArrayLike) -> List[List[int]]:
        """Generate prediction sets."""
        if self._q_hat is None:
            raise NotCalibratedError("AdaptivePredictionSets")

        softmax = np.asarray(softmax_outputs, dtype=np.float64)
        sets = []

        for probs in softmax:
            sorted_idx = np.argsort(-probs)
            sorted_probs = probs[sorted_idx]
            cumsum = np.cumsum(sorted_probs)

            pred_set = []
            for rank, idx in enumerate(sorted_idx):
                penalty = self.lambda_reg * max(0, rank + 1 - self.k_reg)
                threshold = cumsum[rank] - sorted_probs[rank] + penalty
                pred_set.append(int(idx))
                if threshold >= self._q_hat:
                    break

            sets.append(pred_set)
        return sets
