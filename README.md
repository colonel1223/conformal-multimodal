# conformal-multimodal

Distribution-free uncertainty quantification for multimodal ML. Finite-sample coverage guarantees with no distributional assumptions.

## Why

Model confidence scores are miscalibrated. A system reporting 94% confidence can be wrong 30%+ of the time. Conformal prediction provides mathematically guaranteed prediction sets: when the method targets 90% coverage, it achieves ≥90% coverage on any distribution, for any model.

This library extends conformal methods to multimodal systems and streaming settings.

## Install

```bash
pip install -e .          # core
pip install -e ".[dev]"   # + pytest, mypy
```

## Modules

| Module | Method | Reference |
|--------|--------|-----------|
| `core/predictor` | Split conformal prediction | Vovk et al. (2005) |
| `core/quantile` | Conformalized quantile regression (CQR) | Romano et al. (2019) |
| `core/classification` | RAPS: regularized adaptive prediction sets | Angelopoulos et al. (2021) |
| `multimodal/fusion` | Per-modality calibration + adaptive fusion | Novel |
| `multimodal/mondrian` | Group-conditional conformal (Mondrian) | Vovk (2012) |
| `online/tracker` | Adaptive conformal inference under shift | Gibbs & Candès (2021) |
| `risk/control` | Learn-then-Test conformal risk control | Angelopoulos et al. (2022) |
| `metrics/calibration` | ECE, Brier score, Brier decomposition | Naeini et al. (2015) |
| `metrics/efficiency` | Set size, conditional coverage, stratified analysis | — |

## Quick start

```python
from conformal import SplitConformalPredictor

cp = SplitConformalPredictor(alpha=0.1)
cp.calibrate(cal_predictions, cal_labels)
intervals = cp.predict(test_predictions)
# intervals[i].lower, intervals[i].upper — guaranteed ≥ 90% coverage
```

```python
from conformal import MultimodalConformalPredictor
from conformal.multimodal import ModalityConfig

mm = MultimodalConformalPredictor(
    alpha=0.1,
    modalities=[ModalityConfig("vision"), ModalityConfig("language")],
    fusion="adaptive",
)
mm.calibrate({"vision": v_cal, "language": l_cal}, {"vision": y, "language": y})
```

```python
from conformal import OnlineConformalTracker

tracker = OnlineConformalTracker(alpha=0.1, gamma=0.005)
for score, y, pred in stream:
    q_t, covered = tracker.update(score, y, pred)
# Maintains coverage even under distribution shift
```

## Test

```bash
make test          # pytest
make typecheck     # mypy
make benchmark     # coverage-efficiency tables
```

Coverage guarantees verified under Gaussian, Cauchy (heavy-tailed), exponential, and mixture distributions with statistical hypothesis testing at significance level 0.01.

## References

1. Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World.*
2. Romano, Patterson, Candès (2019). *Conformalized Quantile Regression.* NeurIPS.
3. Angelopoulos, Bates, Malik, Jordan (2021). *Uncertainty Sets for Image Classifiers.* ICLR.
4. Vovk (2012). *Conditional Validity of Inductive Conformal Predictors.* AISTATS.
5. Gibbs, Candès (2021). *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS.
6. Angelopoulos, Bates, Fisch, Lei, Schuster (2022). *Conformal Risk Control.* ICLR.

## License

MIT
