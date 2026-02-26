"""Prediction set efficiency and conditional coverage metrics."""

import numpy as np
from typing import Dict, Hashable


def average_set_size(prediction_sets: list) -> float:
    """Mean width (regression) or cardinality (classification) of prediction sets."""
    sizes = []
    for ps in prediction_sets:
        if hasattr(ps, 'width'):
            sizes.append(ps.width)
        elif isinstance(ps, tuple) and len(ps) == 2 and isinstance(ps[0], (int, float)):
            sizes.append(ps[1] - ps[0])
        elif isinstance(ps, (list, set)):
            sizes.append(len(ps))
    return float(np.mean(sizes)) if sizes else 0.0


def conditional_coverage(prediction_sets: list, true_values: np.ndarray,
                         groups: np.ndarray) -> Dict[Hashable, float]:
    """Coverage per subgroup."""
    results = {}
    for g in np.unique(groups):
        mask = groups == g
        covered = 0
        for i in np.where(mask)[0]:
            ps = prediction_sets[i]
            v = true_values[i]
            if hasattr(ps, 'contains'):
                covered += int(ps.contains(v))
            elif isinstance(ps, tuple) and len(ps) == 2:
                covered += int(ps[0] <= v <= ps[1])
            elif isinstance(ps, list):
                covered += int(v in ps)
        results[g] = covered / mask.sum() if mask.sum() > 0 else 0.0
    return results


def size_stratified_coverage(prediction_sets: list, true_values: np.ndarray,
                              n_strata: int = 5) -> Dict[str, Dict]:
    """Coverage stratified by set size — diagnostic for adaptivity."""
    sizes = np.array([
        ps.width if hasattr(ps, 'width') else
        (ps[1]-ps[0]) if isinstance(ps, tuple) else len(ps)
        for ps in prediction_sets
    ])
    quantiles = np.quantile(sizes, np.linspace(0, 1, n_strata + 1))
    results = {}
    for i in range(n_strata):
        mask = (sizes >= quantiles[i]) & (sizes < quantiles[i+1] + 1e-10)
        if mask.sum() == 0: continue
        covered = 0
        for j in np.where(mask)[0]:
            ps = prediction_sets[j]
            v = true_values[j]
            if hasattr(ps, 'contains'): covered += int(ps.contains(v))
            elif isinstance(ps, tuple): covered += int(ps[0] <= v <= ps[1])
        results[f"stratum_{i}"] = {
            "coverage": covered / mask.sum(), "mean_size": float(sizes[mask].mean()),
            "count": int(mask.sum()),
        }
    return results
