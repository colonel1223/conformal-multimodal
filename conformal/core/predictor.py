"""
Split conformal prediction with exchangeability-based coverage guarantees.

Given exchangeable calibration data (X_1,Y_1),...,(X_n,Y_n) and test point
(X_{n+1}, Y_{n+1}), the prediction set

    C_n(X_{n+1}) = {y : s(X_{n+1}, y) <= q_hat}

where q_hat = Quantile(s_1,...,s_n; ceil((n+1)(1-α))/n), satisfies:

    P(Y_{n+1} ∈ C_n(X_{n+1})) >= 1 - α

This holds for ANY joint distribution P_{XY} and ANY score function s,
requiring only exchangeability. The finite-sample correction ceil((n+1)(1-α))/n
accounts for the +1 from the test point [1, Theorem 2.2].

References
----------
[1] Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World.
[2] Angelopoulos, Bates (2021). A Gentle Introduction to Conformal Prediction.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Union, Callable, Sequence
from dataclasses import dataclass
from ..exceptions import NotCalibratedError, InsufficientDataError


@dataclass(frozen=True)
class PredictionInterval:
    """Immutable prediction interval with coverage metadata."""
    lower: float
    upper: float
    confidence: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @staticmethod
    def from_center(center: float, radius: float, confidence: float) -> PredictionInterval:
        return PredictionInterval(
            lower=center - radius, upper=center + radius, confidence=confidence
        )

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


class SplitConformalPredictor:
    r"""Split conformal predictor with finite-sample marginal coverage.

    For miscoverage level α ∈ (0,1), produces prediction sets C_n satisfying:

        P(Y_{n+1} ∈ C_n(X_{n+1})) ≥ 1 - α

    The guarantee is distribution-free and model-agnostic. It requires only
    that calibration + test data are exchangeable (which is implied by i.i.d.).

    Parameters
    ----------
    alpha : float
        Target miscoverage rate. Coverage guarantee is ≥ 1 - alpha.
    score_fn : str or callable
        Nonconformity score function s(ŷ, y). Higher = worse fit.
        Built-in: "absolute" (|ŷ-y|), "squared" ((ŷ-y)²), "softmax" (1-f(x)_y).
        Or callable: (predictions, labels) -> scores array.
    """

    _BUILTIN_SCORES = {
        "absolute": lambda pred, true: np.abs(pred - true),
        "squared": lambda pred, true: (pred - true) ** 2,
        "softmax": lambda pred, true: 1.0 - np.array(
            [pred[i, int(true[i])] for i in range(len(true))]
        ),
    }

    def __init__(self, alpha: float = 0.1,
                 score_fn: Union[str, Callable] = "absolute") -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")

        self.alpha = alpha
        self._q_hat: Optional[float] = None
        self._cal_scores: Optional[np.ndarray] = None
        self._n_cal: int = 0

        if isinstance(score_fn, str):
            if score_fn not in self._BUILTIN_SCORES:
                raise ValueError(
                    f"Unknown score '{score_fn}'. "
                    f"Available: {list(self._BUILTIN_SCORES)}"
                )
            self._score_fn = self._BUILTIN_SCORES[score_fn]
            self._score_name = score_fn
        else:
            self._score_fn = score_fn
            self._score_name = "custom"

    def calibrate(self, predictions: ArrayLike, labels: ArrayLike) -> float:
        r"""Compute conformal quantile from held-out calibration data.

        Sets q̂ = Quantile(s_1,...,s_n; level) where
        level = ⌈(n+1)(1-α)⌉ / n.

        Parameters
        ----------
        predictions : array-like, shape (n,) or (n, n_classes)
            Model predictions on calibration set.
        labels : array-like, shape (n,)
            True labels/values for calibration set.

        Returns
        -------
        q_hat : float
            Calibrated threshold. Scores ≤ q_hat are "conforming."

        Raises
        ------
        InsufficientDataError
            If n < ⌈1/α⌉ - 1 (cannot achieve target coverage).
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        scores = self._score_fn(predictions, labels)
        n = len(scores)

        # Minimum calibration size for meaningful guarantee
        min_n = int(np.ceil(1.0 / self.alpha)) - 1
        if n < min_n:
            raise InsufficientDataError(n, min_n)

        # Finite-sample corrected quantile: ⌈(n+1)(1-α)⌉ / n
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)

        self._q_hat = float(np.quantile(scores, level, method="higher"))
        self._cal_scores = scores.copy()
        self._n_cal = n
        return self._q_hat

    def predict(self, predictions: ArrayLike,
                candidates: Optional[ArrayLike] = None) -> list:
        """Generate prediction sets with coverage guarantee.

        Regression (candidates=None): returns PredictionInterval list.
        Classification (candidates given): returns list of label sets.
        """
        if self._q_hat is None:
            raise NotCalibratedError("SplitConformalPredictor")

        predictions = np.asarray(predictions, dtype=np.float64)

        if candidates is not None:
            return self._predict_classification(predictions, np.asarray(candidates))
        return self._predict_regression(predictions)

    def _predict_regression(self, preds: np.ndarray) -> list[PredictionInterval]:
        """Symmetric intervals: [ŷ - q̂, ŷ + q̂]."""
        return [PredictionInterval.from_center(float(p), self._q_hat, 1 - self.alpha)
                for p in preds]

    def _predict_classification(self, preds: np.ndarray,
                                candidates: np.ndarray) -> list[list]:
        """Prediction sets: {y : s(ŷ, y) ≤ q̂}."""
        sets = []
        for pred in preds:
            if pred.ndim > 0:
                tile = np.tile(pred, (len(candidates), 1))
            else:
                tile = np.full(len(candidates), pred)
            scores = self._score_fn(tile, candidates)
            sets.append(candidates[scores <= self._q_hat].tolist())
        return sets

    def coverage_on(self, predictions: ArrayLike, labels: ArrayLike) -> float:
        """Empirical coverage on held-out test data."""
        if self._q_hat is None:
            raise NotCalibratedError("SplitConformalPredictor")
        scores = self._score_fn(
            np.asarray(predictions, dtype=np.float64),
            np.asarray(labels, dtype=np.float64),
        )
        return float(np.mean(scores <= self._q_hat))

    @property
    def q_hat(self) -> Optional[float]:
        """Calibrated quantile threshold, or None if not calibrated."""
        return self._q_hat

    @property
    def calibration_scores(self) -> Optional[np.ndarray]:
        """Copy of calibration nonconformity scores."""
        return self._cal_scores.copy() if self._cal_scores is not None else None

    def __repr__(self) -> str:
        status = f"q̂={self._q_hat:.4f}" if self._q_hat is not None else "not calibrated"
        return (f"SplitConformalPredictor(α={self.alpha}, "
                f"score='{self._score_name}', {status})")
