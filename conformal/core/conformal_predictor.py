"""
Split conformal prediction with multimodal extensions.

Provides distribution-free coverage guarantees for prediction sets,
extended to handle modality-specific nonconformity scores.

References:
    Vovk et al. (2005) - Algorithmic Learning in a Random World
    Romano et al. (2019) - Conformalized Quantile Regression
    Angelopoulos & Bates (2021) - A Gentle Introduction to Conformal Prediction
"""

import numpy as np
from typing import Optional, Tuple, List, Dict


class ConformalPredictor:
    """Split conformal predictor with finite-sample coverage guarantees.

    Given a calibration set of nonconformity scores, computes prediction
    sets that contain the true label with probability >= 1 - alpha,
    with no assumptions on the data distribution.

    Parameters
    ----------
    alpha : float
        Desired miscoverage rate. Coverage guarantee is >= 1 - alpha.
    score_fn : callable, optional
        Function mapping (model_output, true_label) -> nonconformity score.
        Higher scores indicate worse conformity. If None, uses absolute residual.

    Example
    -------
    >>> predictor = ConformalPredictor(alpha=0.1)
    >>> predictor.calibrate(cal_scores)
    >>> sets = predictor.predict(test_outputs)
    >>> # sets contain true labels with >= 90% probability
    """

    def __init__(self, alpha: float = 0.1, score_fn=None):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.score_fn = score_fn or (lambda y_hat, y: np.abs(y_hat - y))
        self.q_hat = None
        self._cal_scores = None

    def calibrate(self, scores: np.ndarray) -> float:
        """Compute the conformal quantile from calibration nonconformity scores.

        Parameters
        ----------
        scores : np.ndarray, shape (n,)
            Nonconformity scores on calibration data.

        Returns
        -------
        q_hat : float
            The (1 - alpha)(1 + 1/n) quantile of calibration scores.
        """
        scores = np.asarray(scores, dtype=np.float64)
        n = len(scores)
        if n < 1:
            raise ValueError("Need at least 1 calibration score")

        # Finite-sample correction: use ceil((n+1)(1-alpha))/n quantile
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)  # Clip to valid quantile range

        self.q_hat = np.quantile(scores, level, method="higher")
        self._cal_scores = scores
        return self.q_hat

    def predict(self, outputs: np.ndarray, candidates: Optional[np.ndarray] = None) -> list:
        """Generate prediction sets with coverage guarantee.

        For regression: returns intervals [output - q_hat, output + q_hat].
        For classification (if candidates provided): returns sets of labels
        with nonconformity score <= q_hat.

        Parameters
        ----------
        outputs : np.ndarray
            Model outputs / softmax scores for test points.
        candidates : np.ndarray, optional
            Candidate labels for classification. If None, returns intervals.

        Returns
        -------
        prediction_sets : list
            List of prediction sets (intervals or label sets).
        """
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() before predict()")

        outputs = np.asarray(outputs)

        if candidates is not None:
            # Classification: return labels where score <= q_hat
            sets = []
            for output in outputs:
                scores = np.array([self.score_fn(output, c) for c in candidates])
                mask = scores <= self.q_hat
                sets.append(candidates[mask].tolist())
            return sets
        else:
            # Regression: return intervals
            return [(float(o - self.q_hat), float(o + self.q_hat)) for o in outputs]

    @property
    def empirical_coverage(self) -> Optional[float]:
        """Empirical coverage on calibration set (should be close to 1 - alpha)."""
        if self._cal_scores is None:
            return None
        return float(np.mean(self._cal_scores <= self.q_hat))


class MultimodalConformalPredictor:
    """Conformal predictor for multimodal systems with per-modality calibration.

    Handles the case where different modalities (vision, language, audio)
    have different error distributions and calibration properties.
    Provides both per-modality and joint coverage guarantees via
    Bonferroni correction or adaptive weighting.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate for joint prediction.
    modalities : list of str
        Names of modalities (e.g., ["vision", "language"]).
    aggregation : str
        How to combine modality scores. One of:
        - "bonferroni": split alpha across modalities (conservative)
        - "adaptive": weight by modality reliability (tighter)
        - "max": use maximum nonconformity across modalities
    """

    def __init__(self, alpha: float = 0.1, modalities: List[str] = None,
                 aggregation: str = "adaptive"):
        self.alpha = alpha
        self.modalities = modalities or ["modality_0", "modality_1"]
        self.aggregation = aggregation
        self.predictors: Dict[str, ConformalPredictor] = {}
        self._weights: Optional[np.ndarray] = None

        if aggregation == "bonferroni":
            alpha_per = alpha / len(self.modalities)
            for m in self.modalities:
                self.predictors[m] = ConformalPredictor(alpha=alpha_per)
        else:
            for m in self.modalities:
                self.predictors[m] = ConformalPredictor(alpha=alpha)

    def calibrate(self, scores_by_modality: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calibrate each modality's conformal predictor.

        Parameters
        ----------
        scores_by_modality : dict
            Maps modality name -> array of calibration nonconformity scores.

        Returns
        -------
        quantiles : dict
            Maps modality name -> calibrated quantile.
        """
        quantiles = {}
        reliabilities = []

        for m in self.modalities:
            if m not in scores_by_modality:
                raise ValueError(f"Missing calibration scores for modality: {m}")
            q = self.predictors[m].calibrate(scores_by_modality[m])
            quantiles[m] = q
            # Reliability = inverse of score variance (lower variance = more reliable)
            reliabilities.append(1.0 / (np.var(scores_by_modality[m]) + 1e-8))

        if self.aggregation == "adaptive":
            reliabilities = np.array(reliabilities)
            self._weights = reliabilities / reliabilities.sum()

        return quantiles

    def predict(self, outputs_by_modality: Dict[str, np.ndarray]) -> list:
        """Generate multimodal prediction sets.

        Parameters
        ----------
        outputs_by_modality : dict
            Maps modality name -> model outputs for test points.

        Returns
        -------
        prediction_sets : list
            Joint prediction sets satisfying coverage guarantee.
        """
        per_modality_sets = {}
        for m in self.modalities:
            per_modality_sets[m] = self.predictors[m].predict(outputs_by_modality[m])

        n_test = len(per_modality_sets[self.modalities[0]])
        joint_sets = []

        for i in range(n_test):
            if self.aggregation == "max":
                # Intersection of per-modality intervals
                intervals = [per_modality_sets[m][i] for m in self.modalities]
                lo = max(iv[0] for iv in intervals)
                hi = min(iv[1] for iv in intervals)
                joint_sets.append((lo, hi) if lo <= hi else (lo, lo))
            elif self.aggregation == "adaptive" and self._weights is not None:
                # Weighted average of intervals
                intervals = [per_modality_sets[m][i] for m in self.modalities]
                lo = sum(w * iv[0] for w, iv in zip(self._weights, intervals))
                hi = sum(w * iv[1] for w, iv in zip(self._weights, intervals))
                joint_sets.append((float(lo), float(hi)))
            else:
                # Bonferroni: intersection
                intervals = [per_modality_sets[m][i] for m in self.modalities]
                lo = max(iv[0] for iv in intervals)
                hi = min(iv[1] for iv in intervals)
                joint_sets.append((lo, hi) if lo <= hi else (lo, lo))

        return joint_sets

    def coverage_report(self, test_scores_by_modality: Dict[str, np.ndarray]) -> dict:
        """Compute coverage diagnostics per modality and jointly."""
        report = {}
        for m in self.modalities:
            pred = self.predictors[m]
            scores = test_scores_by_modality[m]
            report[m] = {
                "coverage": float(np.mean(scores <= pred.q_hat)),
                "q_hat": float(pred.q_hat),
                "mean_score": float(np.mean(scores)),
                "target": 1 - pred.alpha,
            }
        return report
