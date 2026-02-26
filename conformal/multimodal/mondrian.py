"""
Mondrian conformal prediction: group-conditional coverage.

Standard conformal guarantees marginal coverage: averaged over the entire
test distribution. But this allows undercoverage on subgroups. Mondrian
conformal calibrates SEPARATELY within each group, guaranteeing:

    P(Y ∈ C(X) | G = g) ≥ 1 - α    for each group g

This is critical for fairness: a medical model shouldn't have 95%
coverage overall but only 70% for a specific demographic.

References
----------
[1] Vovk (2012). "Conditional Validity of Inductive Conformal Predictors."
    AISTATS.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Optional, Hashable
from ..core.predictor import SplitConformalPredictor
from ..exceptions import NotCalibratedError


class MondrianConformalPredictor:
    """Per-group conformal predictor with conditional coverage guarantees.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate, applied per group.
    score_fn : str or callable
        Nonconformity score function, passed to SplitConformalPredictor.
    min_group_size : int
        Minimum calibration points per group. Groups with fewer points
        fall back to the global predictor.
    """

    def __init__(self, alpha: float = 0.1, score_fn: str = "absolute",
                 min_group_size: int = 30) -> None:
        self.alpha = alpha
        self.score_fn = score_fn
        self.min_group_size = min_group_size
        self._group_predictors: Dict[Hashable, SplitConformalPredictor] = {}
        self._fallback: Optional[SplitConformalPredictor] = None
        self._group_sizes: Dict[Hashable, int] = {}

    def calibrate(self, predictions: ArrayLike, labels: ArrayLike,
                  groups: ArrayLike) -> Dict[Hashable, float]:
        """Calibrate per group.

        Parameters
        ----------
        predictions, labels : array-like, shape (n,)
        groups : array-like, shape (n,)
            Group membership indicators (any hashable type).

        Returns
        -------
        quantiles : dict
            {group: q_hat} for each group.
        """
        preds = np.asarray(predictions, dtype=np.float64)
        labs = np.asarray(labels, dtype=np.float64)
        grps = np.asarray(groups)

        # Fallback predictor uses all data
        self._fallback = SplitConformalPredictor(alpha=self.alpha, score_fn=self.score_fn)
        self._fallback.calibrate(preds, labs)

        quantiles = {}
        for g in np.unique(grps):
            mask = grps == g
            n_g = int(mask.sum())
            self._group_sizes[g] = n_g

            if n_g >= self.min_group_size:
                cp = SplitConformalPredictor(alpha=self.alpha, score_fn=self.score_fn)
                try:
                    q = cp.calibrate(preds[mask], labs[mask])
                    self._group_predictors[g] = cp
                    quantiles[g] = q
                except Exception:
                    quantiles[g] = self._fallback.q_hat
            else:
                quantiles[g] = self._fallback.q_hat

        return quantiles

    def predict(self, predictions: ArrayLike, groups: ArrayLike) -> list:
        """Generate group-conditional prediction sets."""
        if self._fallback is None:
            raise NotCalibratedError("MondrianConformalPredictor")

        preds = np.asarray(predictions, dtype=np.float64)
        grps = np.asarray(groups)
        results = []

        for i in range(len(preds)):
            g = grps[i]
            cp = self._group_predictors.get(g, self._fallback)
            result = cp.predict(preds[i:i+1])
            results.append(result[0])

        return results

    def conditional_coverage(self, predictions: ArrayLike, labels: ArrayLike,
                              groups: ArrayLike) -> Dict[Hashable, float]:
        """Compute coverage per group on test data."""
        preds = np.asarray(predictions, dtype=np.float64)
        labs = np.asarray(labels, dtype=np.float64)
        grps = np.asarray(groups)

        coverages = {}
        for g in np.unique(grps):
            mask = grps == g
            cp = self._group_predictors.get(g, self._fallback)
            coverages[g] = cp.coverage_on(preds[mask], labs[mask])
        return coverages
