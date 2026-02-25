# conformal-multimodal

**Teaching machines to say "I don't know."**

## The problem

A vision-language model tells you it's 94% confident. It's wrong. This happens constantly, and it's not a minor inconvenience — it's a structural failure that kills people in medical imaging, autonomous driving, and anywhere else confidence scores get treated as probabilities.

Conformal prediction fixes this. It provides distribution-free, finite-sample coverage guarantees — meaning when the system says "I'm 90% sure," it's actually right at least 90% of the time. No assumptions about the model. No assumptions about the data distribution. Math, not hope.

## What this project does

Extends conformal methods to multimodal settings where calibration is hardest — vision-language, audio-text — because each modality fails differently and the joint distribution is worse than either alone.

## Methods

- Split conformal prediction with modality-aware nonconformity scores
- Adaptive prediction sets that tighten under high cross-modal agreement
- Per-modality and joint coverage guarantees

## Usage

```bash
pip install -r requirements.txt
python calibrate.py --model clip --data val_set/ --alpha 0.1
```

## Requirements

Python 3.9+, PyTorch, NumPy, SciPy

## License

MIT
