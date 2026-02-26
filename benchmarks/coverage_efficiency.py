"""Benchmark: coverage vs interval width across distributions, alphas, and calibration set sizes."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conformal.core.predictor import SplitConformalPredictor

def run():
    rng = np.random.default_rng(0)
    dists = {
        "gaussian_0.3":  lambda n: rng.standard_normal(n) * 0.3,
        "cauchy_0.05":   lambda n: rng.standard_cauchy(n) * 0.05,
        "uniform_0.5":   lambda n: rng.uniform(-0.5, 0.5, n),
        "mixture":       lambda n: np.where(rng.random(n) > 0.85,
                                             rng.standard_normal(n) * 2.0,
                                             rng.standard_normal(n) * 0.1),
    }
    cal_sizes = [100, 500, 2000]

    print(f"{'Dist':<16} {'n_cal':>6} {'α':>5} {'Coverage':>9} {'Width':>8} {'Valid':>6}")
    print("-" * 55)

    for name, noise_fn in dists.items():
        for n_cal in cal_sizes:
            for alpha in [0.05, 0.1, 0.2]:
                n_test = 5000
                n = n_cal + n_test
                y = rng.standard_normal(n)
                pred = y + noise_fn(n)
                try:
                    cp = SplitConformalPredictor(alpha=alpha)
                    cp.calibrate(pred[:n_cal], y[:n_cal])
                    intervals = cp.predict(pred[n_cal:])
                    cov = sum(1 for i, iv in enumerate(intervals) if iv.contains(y[n_cal+i])) / n_test
                    width = np.mean([iv.width for iv in intervals])
                    valid = "✓" if cov >= (1-alpha) - 0.05 else "✗"
                    print(f"{name:<16} {n_cal:>6} {alpha:>5.2f} {cov:>9.4f} {width:>8.3f} {valid:>6}")
                except Exception as e:
                    print(f"{name:<16} {n_cal:>6} {alpha:>5.2f}  — {e}")

if __name__ == "__main__":
    run()
