# conformal-multimodal

**Teaching machines to say "I don't know."**

Distribution-free uncertainty quantification for multimodal ML systems. When a model says 90% confident, conformal prediction guarantees it's right at least 90% of the time — no assumptions about the data distribution.

## Why this matters

Neural network confidence scores are miscalibrated. A vision-language model reporting 94% confidence can be wrong 30%+ of the time. Conformal prediction provides mathematically guaranteed coverage without distributional assumptions. This library extends those guarantees to multimodal settings where each modality fails differently.

## Install

```bash
pip install -e .
```

## Quick start

```python
from conformal.core import ConformalPredictor, MultimodalConformalPredictor

# Single-modality
predictor = ConformalPredictor(alpha=0.1)
predictor.calibrate(calibration_scores)
prediction_sets = predictor.predict(test_outputs)

# Multimodal with per-modality calibration
mm = MultimodalConformalPredictor(
    alpha=0.1,
    modalities=["vision", "language"],
    aggregation="adaptive"  # weights by modality reliability
)
mm.calibrate({"vision": vis_scores, "language": lang_scores})
joint_sets = mm.predict({"vision": vis_out, "language": lang_out})
```

## What's implemented

- `ConformalPredictor` — Split conformal with finite-sample correction. Classification and regression.
- `MultimodalConformalPredictor` — Per-modality calibration with Bonferroni, adaptive, or max aggregation.
- `calibration_error` — Expected Calibration Error (ECE) with binning.
- `brier_score` — Probabilistic prediction quality.
- `coverage_diagnostic` — Empirical coverage and set size analysis.

## Tests

```bash
cd tests && python test_conformal.py
```

## Structure

```
├── conformal/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── conformal_predictor.py   # Core: ConformalPredictor, MultimodalConformalPredictor
│       └── calibration.py           # ECE, Brier score, coverage diagnostics
├── tests/
│   └── test_conformal.py            # Coverage guarantee verification
├── docs/
│   └── index.html                   # Documentation site
├── setup.py
├── requirements.txt
└── README.md
```

## References

- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World*
- Romano, Patterson & Candès (2019). *Conformalized Quantile Regression*
- Angelopoulos & Bates (2021). *A Gentle Introduction to Conformal Prediction*

## License

MIT
