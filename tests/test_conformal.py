"""Tests verifying coverage guarantees hold empirically."""

import numpy as np
import sys
sys.path.insert(0, '..')
from conformal.core.conformal_predictor import ConformalPredictor, MultimodalConformalPredictor
from conformal.core.calibration import calibration_error, brier_score, coverage_diagnostic


def test_coverage_guarantee():
    """Verify that conformal prediction achieves target coverage."""
    np.random.seed(42)
    n_cal, n_test = 500, 1000
    alpha = 0.1

    # Generate synthetic data
    true_values = np.random.randn(n_cal + n_test)
    noise = np.random.randn(n_cal + n_test) * 0.5
    predictions = true_values + noise

    cal_scores = np.abs(predictions[:n_cal] - true_values[:n_cal])
    test_predictions = predictions[n_cal:]
    test_true = true_values[n_cal:]

    # Calibrate and predict
    predictor = ConformalPredictor(alpha=alpha)
    predictor.calibrate(cal_scores)
    intervals = predictor.predict(test_predictions)

    # Check coverage
    covered = sum(1 for i, (lo, hi) in enumerate(intervals)
                  if lo <= test_true[i] <= hi)
    coverage = covered / n_test

    # Coverage should be >= 1 - alpha (with high probability)
    assert coverage >= (1 - alpha) - 0.05, \
        f"Coverage {coverage:.3f} below target {1 - alpha}"
    print(f"  Coverage: {coverage:.3f} (target: {1 - alpha})")


def test_multimodal_coverage():
    """Verify joint coverage across modalities."""
    np.random.seed(42)
    n_cal, n_test = 500, 1000
    alpha = 0.1

    predictor = MultimodalConformalPredictor(
        alpha=alpha,
        modalities=["vision", "language"],
        aggregation="bonferroni"
    )

    # Calibrate with different noise levels per modality
    cal_scores = {
        "vision": np.abs(np.random.randn(n_cal) * 0.3),
        "language": np.abs(np.random.randn(n_cal) * 0.7),
    }
    predictor.calibrate(cal_scores)

    report = predictor.coverage_report({
        "vision": np.abs(np.random.randn(n_test) * 0.3),
        "language": np.abs(np.random.randn(n_test) * 0.7),
    })

    for m, r in report.items():
        print(f"  {m}: coverage={r['coverage']:.3f} (target={r['target']:.3f})")
        assert r["coverage"] >= r["target"] - 0.05


def test_calibration_metrics():
    """Test ECE and Brier score computation."""
    # Perfectly calibrated
    confs = np.array([0.9, 0.9, 0.9, 0.1, 0.1])
    accs = np.array([1, 1, 1, 0, 0])
    ece, _, _ = calibration_error(confs, accs, n_bins=10)
    assert ece < 0.15, f"ECE too high for calibrated predictions: {ece}"

    # Brier score
    bs = brier_score(np.array([0.9, 0.1]), np.array([1, 0]))
    assert bs < 0.02, f"Brier score too high: {bs}"
    print(f"  ECE: {ece:.4f}, Brier: {bs:.4f}")


if __name__ == "__main__":
    print("Running conformal-multimodal tests...\n")
    test_coverage_guarantee()
    test_multimodal_coverage()
    test_calibration_metrics()
    print("\nAll tests passed.")
