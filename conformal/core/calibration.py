"""Calibration metrics for evaluating prediction quality."""

import numpy as np
from typing import Tuple


def calibration_error(confidences: np.ndarray, accuracies: np.ndarray,
                      n_bins: int = 15) -> Tuple[float, np.ndarray, np.ndarray]:
    """Expected Calibration Error (ECE).

    Measures the gap between predicted confidence and actual accuracy
    across binned confidence levels.

    Parameters
    ----------
    confidences : np.ndarray
        Predicted confidence scores in [0, 1].
    accuracies : np.ndarray
        Binary correctness indicators (0 or 1).
    n_bins : int
        Number of bins for grouping predictions.

    Returns
    -------
    ece : float
        Expected calibration error (lower is better, 0 = perfectly calibrated).
    bin_confidences : np.ndarray
        Mean confidence per bin.
    bin_accuracies : np.ndarray
        Mean accuracy per bin.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_confidences[i] = confidences[mask].mean()
            bin_accuracies[i] = accuracies[mask].mean()
            bin_counts[i] = mask.sum()

    total = bin_counts.sum()
    if total == 0:
        return 0.0, bin_confidences, bin_accuracies

    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / total
    return float(ece), bin_confidences, bin_accuracies


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score for probabilistic predictions.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted probabilities in [0, 1].
    outcomes : np.ndarray
        Binary outcomes (0 or 1).

    Returns
    -------
    score : float
        Mean squared error between predictions and outcomes.
        0 = perfect, 1 = worst possible.
    """
    return float(np.mean((np.asarray(probabilities) - np.asarray(outcomes)) ** 2))


def coverage_diagnostic(prediction_sets: list, true_values: np.ndarray) -> dict:
    """Evaluate coverage and efficiency of conformal prediction sets.

    Parameters
    ----------
    prediction_sets : list
        List of (lower, upper) intervals or label sets.
    true_values : np.ndarray
        True labels/values.

    Returns
    -------
    diagnostics : dict
        Coverage rate, average set size, and conditional coverage breakdown.
    """
    true_values = np.asarray(true_values)
    n = len(true_values)

    covered = 0
    total_size = 0.0

    for i in range(n):
        pset = prediction_sets[i]
        if isinstance(pset, (tuple, list)) and len(pset) == 2:
            # Interval
            if pset[0] <= true_values[i] <= pset[1]:
                covered += 1
            total_size += pset[1] - pset[0]
        elif isinstance(pset, (list, set)):
            # Label set
            if true_values[i] in pset:
                covered += 1
            total_size += len(pset)

    return {
        "coverage": covered / n if n > 0 else 0.0,
        "avg_set_size": total_size / n if n > 0 else 0.0,
        "n_samples": n,
    }
