"""
Conformalized Quantile Regression (CQR).

Combines quantile regression with conformal calibration. The base model
predicts conditional quantiles q̂_lo(x) ≈ Q_{α/2}(Y|X=x) and
q̂_hi(x) ≈ Q_{1-α/2}(Y|X=x). CQR then calibrates these using:

    s_i = max(q̂_lo(X_i) - Y_i, Y_i - q̂_hi(X_i))

yielding intervals [q̂_lo(x) - q̂, q̂_hi(x) + q̂] that are both:
- Adaptive: width varies with model's estimated uncertainty
- Valid: marginal coverage ≥ 1-α regardless of quantile model quality

Key insight: even if the quantile model is badly miscalibrated, CQR
corrects for it. If the model is good, CQR intervals are tight. If
bad, they're wide but still valid. You never lose coverage.

References
----------
[1] Romano, Patterson, Candès. "Conformalized Quantile Regression."
    NeurIPS, 2019.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, List
from ..exceptions import NotCalibratedError, InsufficientDataError


class ConformalizedQuantileRegressor:
    """CQR: adaptive intervals with distribution-free coverage.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    symmetric : bool
        If True, use symmetric scores max(lo-y, y-hi). If False,
        use asymmetric correction (different adjustments for lo/hi).
    """

    def __init__(self, alpha: float = 0.1, symmetric: bool = True) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
        self.symmetric = symmetric
        self._q_hat: Optional[float] = None
        self._q_lo_hat: Optional[float] = None
        self._q_hi_hat: Optional[float] = None

    def calibrate(self, quantile_lo: ArrayLike, quantile_hi: ArrayLike,
                  labels: ArrayLike) -> float:
        """Calibrate using held-out data with pre-computed quantile predictions.

        Parameters
        ----------
        quantile_lo, quantile_hi : array-like, shape (n,)
            Lower/upper quantile predictions from base model.
        labels : array-like, shape (n,)
            True values.

        Returns
        -------
        q_hat : float
            Conformal correction term.
        """
        lo = np.asarray(quantile_lo, dtype=np.float64)
        hi = np.asarray(quantile_hi, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        n = len(y)

        min_n = int(np.ceil(1.0 / self.alpha)) - 1
        if n < min_n:
            raise InsufficientDataError(n, min_n)

        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)

        if self.symmetric:
            scores = np.maximum(lo - y, y - hi)
            self._q_hat = float(np.quantile(scores, level, method="higher"))
        else:
            scores_lo = lo - y  # Positive when lo overestimates
            scores_hi = y - hi  # Positive when hi underestimates
            self._q_lo_hat = float(np.quantile(scores_lo, level, method="higher"))
            self._q_hi_hat = float(np.quantile(scores_hi, level, method="higher"))
            self._q_hat = max(self._q_lo_hat, self._q_hi_hat)

        return self._q_hat

    def predict(self, quantile_lo: ArrayLike,
                quantile_hi: ArrayLike) -> List[Tuple[float, float]]:
        """Produce calibrated prediction intervals.

        Returns list of (lower, upper) tuples.
        """
        if self._q_hat is None:
            raise NotCalibratedError("ConformalizedQuantileRegressor")

        lo = np.asarray(quantile_lo, dtype=np.float64)
        hi = np.asarray(quantile_hi, dtype=np.float64)

        if self.symmetric:
            return [(float(l - self._q_hat), float(h + self._q_hat))
                    for l, h in zip(lo, hi)]
        else:
            return [(float(l - self._q_lo_hat), float(h + self._q_hi_hat))
                    for l, h in zip(lo, hi)]
