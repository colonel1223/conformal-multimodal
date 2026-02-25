# conformal-multimodal

Conformal prediction methods for provably-calibrated uncertainty quantification in multimodal ML systems.

## Overview

Standard neural networks produce confidence scores that are poorly calibrated — a model saying "90% confident" is often wrong far more than 10% of the time. Conformal prediction provides distribution-free, finite-sample coverage guarantees regardless of the underlying model.

This project extends conformal methods to multimodal settings (vision-language, audio-text) where calibration is particularly challenging due to modality-specific error distributions.

## Methods

- Split conformal prediction with modality-aware nonconformity scores
- Adaptive prediction sets that tighten under high cross-modal agreement
- Coverage guarantees that hold per-modality and jointly

## Usage

```bash
pip install -r requirements.txt
python calibrate.py --model clip --data val_set/ --alpha 0.1
```

## Requirements

- Python 3.9+
- PyTorch
- NumPy, SciPy

## License

MIT
