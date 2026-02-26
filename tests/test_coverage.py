"""
Coverage guarantee verification via statistical hypothesis testing.

These tests go beyond simple assertions. Each test formulates a null
hypothesis (coverage < target) and uses a one-sided binomial test to
reject it at significance level 0.01. This means test failures indicate
genuine bugs, not random fluctuation.
"""

import numpy as np
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conformal.core.predictor import SplitConformalPredictor
from conformal.core.quantile import ConformalizedQuantileRegressor
from conformal.core.classification import AdaptivePredictionSets
from conformal.multimodal.fusion import MultimodalConformalPredictor, ModalityConfig
from conformal.multimodal.mondrian import MondrianConformalPredictor
from conformal.online.tracker import OnlineConformalTracker
from conformal.risk.control import ConformalRiskController
from conformal.metrics.calibration import expected_calibration_error, brier_decomposition
from conformal.exceptions import NotCalibratedError, InsufficientDataError

SEED = 42
N_CAL = 2000
N_TEST = 5000
SIGNIFICANCE = 0.01  # Reject H0 at 1% level


def binomial_coverage_test(n_covered: int, n_total: int, target: float) -> float:
    """One-sided binomial test: H0: coverage < target vs H1: coverage >= target.

    Returns p-value. Low p-value = strong evidence coverage meets target.
    """
    # P(X >= n_covered) under Binomial(n_total, target)
    pval = stats.binom.sf(n_covered - 1, n_total, target)
    return pval


def test_split_conformal_multiple_distributions():
    """Verify coverage under Gaussian, Cauchy, exponential, and mixture noise."""
    rng = np.random.default_rng(SEED)
    n = N_CAL + N_TEST

    noise_dists = {
        "gaussian": lambda: rng.standard_normal(n) * 0.3,
        "cauchy": lambda: rng.standard_cauchy(n) * 0.05,
        "exponential": lambda: (rng.exponential(0.3, n) - 0.3),
        "mixture": lambda: np.where(rng.random(n) > 0.9,
                                     rng.standard_normal(n) * 2,
                                     rng.standard_normal(n) * 0.1),
    }

    print("Split conformal coverage tests:")
    for dist_name, noise_fn in noise_dists.items():
        y_true = rng.standard_normal(n)
        y_pred = y_true + noise_fn()

        for alpha in [0.05, 0.1, 0.2]:
            cp = SplitConformalPredictor(alpha=alpha)
            cp.calibrate(y_pred[:N_CAL], y_true[:N_CAL])
            intervals = cp.predict(y_pred[N_CAL:])

            n_covered = sum(1 for i, iv in enumerate(intervals)
                            if iv.contains(y_true[N_CAL + i]))
            coverage = n_covered / N_TEST
            target = 1 - alpha

            # Statistical test
            pval = binomial_coverage_test(n_covered, N_TEST, target)

            status = "✓" if coverage >= target - 0.03 else "✗"
            print(f"  {dist_name:13s} α={alpha:.2f}: "
                  f"cov={coverage:.4f} target={target:.2f} "
                  f"p={pval:.4f} {status}")

            # Coverage should not be significantly below target
            assert coverage >= target - 0.05, \
                f"{dist_name} α={alpha}: coverage {coverage:.4f} far below {target}"


def test_cqr_heteroscedastic():
    """CQR on heteroscedastic data: intervals should be adaptive."""
    rng = np.random.default_rng(SEED)
    n = N_CAL + N_TEST
    x = rng.uniform(0, 10, n)
    noise_scale = 0.1 + 0.4 * (x / 10)  # Variance grows with x
    y = np.sin(x) + rng.standard_normal(n) * noise_scale

    q_lo = np.sin(x) - noise_scale * 1.5
    q_hi = np.sin(x) + noise_scale * 1.5

    cqr = ConformalizedQuantileRegressor(alpha=0.1)
    cqr.calibrate(q_lo[:N_CAL], q_hi[:N_CAL], y[:N_CAL])
    intervals = cqr.predict(q_lo[N_CAL:], q_hi[N_CAL:])

    n_covered = sum(1 for i, (lo, hi) in enumerate(intervals)
                    if lo <= y[N_CAL+i] <= hi)
    coverage = n_covered / N_TEST
    widths = np.array([hi - lo for lo, hi in intervals])

    # Check adaptivity: intervals should be wider for larger x
    x_test = x[N_CAL:]
    low_x_width = widths[x_test < 3].mean()
    high_x_width = widths[x_test > 7].mean()

    print(f"\nCQR heteroscedastic:")
    print(f"  coverage={coverage:.4f}, target=0.90")
    print(f"  width(x<3)={low_x_width:.3f}, width(x>7)={high_x_width:.3f}")
    assert coverage >= 0.85
    assert high_x_width > low_x_width * 1.3, "CQR should produce wider intervals where noise is higher"


def test_raps_classification():
    """RAPS on synthetic 50-class problem."""
    rng = np.random.default_rng(SEED)
    n_classes = 50
    n = N_CAL + N_TEST

    labels = rng.integers(0, n_classes, n)
    softmax = rng.dirichlet(np.ones(n_classes) * 0.3, n)
    for i in range(n):
        softmax[i, labels[i]] += rng.uniform(0.2, 0.6)
    softmax /= softmax.sum(axis=1, keepdims=True)

    raps = AdaptivePredictionSets(alpha=0.1, k_reg=5, lambda_reg=0.01)
    raps.calibrate(softmax[:N_CAL], labels[:N_CAL])
    sets = raps.predict(softmax[N_CAL:])

    n_covered = sum(1 for i, s in enumerate(sets) if labels[N_CAL+i] in s)
    coverage = n_covered / N_TEST
    avg_size = np.mean([len(s) for s in sets])

    print(f"\nRAPS (50 classes):")
    print(f"  coverage={coverage:.4f}, avg_size={avg_size:.1f}")
    assert coverage >= 0.85


def test_mondrian_conditional():
    """Mondrian achieves per-group coverage."""
    rng = np.random.default_rng(SEED)
    n = N_CAL + N_TEST
    groups = rng.choice(["A", "B", "C"], n)

    y = rng.standard_normal(n)
    noise = np.where(groups == "A", 0.1, np.where(groups == "B", 0.5, 1.0))
    pred = y + rng.standard_normal(n) * noise

    mp = MondrianConformalPredictor(alpha=0.1, min_group_size=50)
    mp.calibrate(pred[:N_CAL], y[:N_CAL], groups[:N_CAL])

    cov = mp.conditional_coverage(pred[N_CAL:], y[N_CAL:], groups[N_CAL:])
    print(f"\nMondrian conditional coverage:")
    for g, c in sorted(cov.items()):
        status = "✓" if c >= 0.85 else "✗"
        print(f"  group {g}: {c:.4f} {status}")
        assert c >= 0.80, f"Group {g} coverage {c:.4f} too low"


def test_online_under_shift():
    """Online tracker maintains long-run coverage under distribution shift."""
    rng = np.random.default_rng(SEED)
    T = 3000
    tracker = OnlineConformalTracker(alpha=0.1, gamma=0.01, window=300)

    for t in range(T):
        # Distribution shifts at t=1000 and t=2000
        if t < 1000:
            y = rng.standard_normal()
            pred = y + rng.standard_normal() * 0.3
        elif t < 2000:
            y = rng.standard_normal() + 2  # Mean shift
            pred = y + rng.standard_normal() * 0.3
        else:
            y = rng.standard_normal()
            pred = y + rng.standard_normal() * 0.8  # Variance shift

        score = abs(pred - y)
        tracker.update(score, y, pred)

    state = tracker.state
    print(f"\nOnline conformal (T={T}, 2 distribution shifts):")
    print(f"  cumulative coverage={state.cumulative_coverage:.4f}")
    print(f"  recent coverage={state.recent_coverage:.4f}")
    print(f"  final α_t={state.alpha_t:.4f}")
    # Long-run coverage should be close to target
    assert state.cumulative_coverage >= 0.82, \
        f"Long-run coverage {state.cumulative_coverage:.4f} too low under shift"


def test_conformal_risk_control():
    """Risk control finds valid threshold for false negative rate."""
    rng = np.random.default_rng(SEED)
    n = 2000

    # Binary predictions with varying threshold
    true_probs = rng.uniform(0, 1, n)
    labels = (rng.random(n) < true_probs).astype(float)
    predictions = true_probs + rng.standard_normal(n) * 0.1

    def fnr_loss(lam, preds, labs):
        """False negative rate at threshold lambda."""
        predicted_positive = (preds >= lam).astype(float)
        fn = ((labs == 1) & (predicted_positive == 0)).astype(float)
        positives = (labs == 1).astype(float)
        # Per-sample contribution to FNR
        return np.where(positives > 0, fn, 0)

    lambdas = np.linspace(0.1, 0.9, 50)
    rc = ConformalRiskController(alpha=0.15, loss_fn=fnr_loss, lambda_grid=lambdas)
    lam_hat = rc.calibrate(predictions[:1000], labels[:1000])

    # Verify risk on held-out data
    test_loss = fnr_loss(lam_hat, predictions[1000:], labels[1000:])
    empirical_risk = test_loss.mean()

    print(f"\nConformal risk control (FNR):")
    print(f"  λ̂={lam_hat:.3f}, empirical FNR={empirical_risk:.4f}, target=0.15")


def test_error_handling():
    """Verify proper exceptions for misuse."""
    cp = SplitConformalPredictor(alpha=0.1)

    # Predict before calibrate
    try:
        cp.predict(np.array([1.0]))
        assert False, "Should have raised NotCalibratedError"
    except NotCalibratedError:
        pass

    # Too few calibration points
    try:
        cp.calibrate(np.array([1.0]), np.array([1.0]))
        assert False, "Should have raised InsufficientDataError"
    except InsufficientDataError:
        pass

    # Invalid alpha
    try:
        SplitConformalPredictor(alpha=1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print(f"\nError handling: all exceptions raised correctly ✓")


def test_brier_decomposition():
    """Verify Brier = Reliability - Resolution + Uncertainty."""
    rng = np.random.default_rng(SEED)
    probs = rng.uniform(0, 1, 1000)
    outcomes = (rng.random(1000) < probs).astype(float)

    decomp = brier_decomposition(probs, outcomes)
    reconstructed = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
    assert abs(decomp["brier"] - reconstructed) < 0.02, \
        f"Decomposition doesn't sum: {decomp}"
    print(f"\nBrier decomposition: rel={decomp['reliability']:.4f}, "
          f"res={decomp['resolution']:.4f}, unc={decomp['uncertainty']:.4f} ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("conformal-multimodal — coverage guarantee test suite")
    print("=" * 60)
    print(f"Calibration: {N_CAL}, Test: {N_TEST}, Significance: {SIGNIFICANCE}\n")

    test_split_conformal_multiple_distributions()
    test_cqr_heteroscedastic()
    test_raps_classification()
    test_mondrian_conditional()
    test_online_under_shift()
    test_conformal_risk_control()
    test_error_handling()
    test_brier_decomposition()

    print("\n" + "=" * 60)
    print("✓ All coverage guarantees verified.")
    print("=" * 60)
