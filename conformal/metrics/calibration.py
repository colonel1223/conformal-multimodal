"""Calibration metrics: ECE, Brier score with decomposition."""

import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, Dict


def expected_calibration_error(confidences: ArrayLike, accuracies: ArrayLike,
                               n_bins: int = 15,
                               norm: str = "l1") -> Tuple[float, Dict]:
    """Expected Calibration Error.

    ECE = Σ_b (n_b/n) |acc(b) - conf(b)|

    Parameters
    ----------
    confidences : shape (n,), values in [0,1]
    accuracies : shape (n,), binary
    n_bins : number of equal-width bins
    norm : "l1" for ECE, "l2" for RMSCE, "max" for MCE

    Returns (ece, bin_data_dict).
    """
    conf = np.asarray(confidences, dtype=np.float64)
    acc = np.asarray(accuracies, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    b_conf, b_acc, b_count = [], [], []

    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i+1])
        c = int(mask.sum())
        b_count.append(c)
        b_conf.append(float(conf[mask].mean()) if c > 0 else 0.0)
        b_acc.append(float(acc[mask].mean()) if c > 0 else 0.0)

    b_count = np.array(b_count); b_conf = np.array(b_conf); b_acc = np.array(b_acc)
    total = b_count.sum()
    if total == 0:
        return 0.0, {}

    gaps = np.abs(b_acc - b_conf)
    if norm == "l1":
        val = float(np.sum(b_count * gaps) / total)
    elif norm == "l2":
        val = float(np.sqrt(np.sum(b_count * gaps**2) / total))
    else:
        val = float(np.max(gaps[b_count > 0])) if np.any(b_count > 0) else 0.0

    return val, {"bin_confidences": b_conf.tolist(), "bin_accuracies": b_acc.tolist(),
                 "bin_counts": b_count.tolist()}


def brier_score(probabilities: ArrayLike, outcomes: ArrayLike) -> float:
    """Brier score: MSE of probabilistic predictions. 0=perfect, 1=worst."""
    return float(np.mean((np.asarray(probabilities) - np.asarray(outcomes))**2))


def brier_decomposition(probabilities: ArrayLike, outcomes: ArrayLike,
                         n_bins: int = 10) -> Dict[str, float]:
    """Murphy decomposition: Brier = Reliability - Resolution + Uncertainty.

    Reliability: how far calibration is from perfect (lower better).
    Resolution: how much predictions deviate from base rate (higher better).
    Uncertainty: entropy of the base rate (fixed for given data).
    """
    p = np.asarray(probabilities, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)
    base_rate = o.mean()

    edges = np.linspace(0, 1, n_bins + 1)
    rel, res = 0.0, 0.0
    n = len(o)

    for i in range(n_bins):
        mask = (p > edges[i]) & (p <= edges[i+1])
        nk = mask.sum()
        if nk == 0: continue
        ok = o[mask].mean()
        pk = p[mask].mean()
        rel += nk * (pk - ok)**2
        res += nk * (ok - base_rate)**2

    rel /= n; res /= n
    unc = base_rate * (1 - base_rate)

    return {"brier": float(brier_score(p, o)), "reliability": float(rel),
            "resolution": float(res), "uncertainty": float(unc)}
