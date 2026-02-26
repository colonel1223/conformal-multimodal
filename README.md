# conformal-multimodal

**Teaching machines to say "I don't know."**

## The problem

A vision-language model says it's 94% confident. It's wrong. Conformal prediction fixes this with distribution-free, finite-sample coverage guarantees — when the system says 90% sure, it's actually right at least 90% of the time.

## What's here

Python library extending conformal methods to multimodal settings where calibration is hardest.

## Structure

```
├── conformal/
│   └── core/
│       └── conformal_predictor.py   # Core prediction engine
├── docs/
│   ├── index.html                   # Documentation site
│   └── .nojekyll
├── setup.py                         # Package installation
└── README.md
```

## Install

```bash
pip install -e .
```

## Usage

```python
from conformal.core.conformal_predictor import ConformalPredictor

predictor = ConformalPredictor(alpha=0.1)
predictor.calibrate(cal_scores)
prediction_set = predictor.predict(new_input)
```

## Methods

- Split conformal prediction with modality-aware nonconformity scores
- Adaptive prediction sets that tighten under cross-modal agreement
- Per-modality and joint coverage guarantees

## Status

Active development. Core predictor functional, extending to vision-language and audio-text.

## License

MIT
