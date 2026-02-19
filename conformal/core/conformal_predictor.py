"""Conformal prediction scaffold with coverage guarantee."""
import numpy as np
import torch

class ConformalPredictor:
    def __init__(self, model, target_coverage=0.90):
        self.model = model
        self.alpha = 1 - target_coverage
        self.threshold = None

    def calibrate(self, cal_scores):
        scores = np.asarray(cal_scores)
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = np.quantile(scores, min(q, 1.0))

    def predict(self, scores_for_labels):
        scores = np.asarray(scores_for_labels)
        return [i for i, s in enumerate(scores) if self.threshold is not None and s <= self.threshold]
