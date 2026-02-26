"""
conformal-multimodal
====================

Distribution-free uncertainty quantification for multimodal ML systems.

Implements split conformal prediction [1], conformalized quantile
regression [2], RAPS [3], Mondrian conformal prediction [4], online
conformal with martingale tracking [5], and Learn-then-Test conformal
risk control [6]. Extended to multimodal settings with per-modality
calibration and adaptive reliability-weighted fusion.

All methods provide finite-sample guarantees under exchangeability
(or weaker conditions for online methods) with no distributional
assumptions on the data-generating process.

References
----------
[1] Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World.
[2] Romano, Patterson, Candès (2019). Conformalized Quantile Regression.
[3] Angelopoulos, Bates, Malik, Jordan (2021). Uncertainty Sets for Image
    Classifiers using Conformal Prediction.
[4] Vovk (2012). Conditional Validity of Inductive Conformal Predictors.
[5] Gibbs, Candès (2021). Adaptive Conformal Inference Under Distribution Shift.
[6] Angelopoulos, Bates, Fisch, Lei, Schuster (2022). Conformal Risk Control.
"""
__version__ = "0.3.0"

from .core.predictor import SplitConformalPredictor
from .core.quantile import ConformalizedQuantileRegressor
from .core.classification import AdaptivePredictionSets
from .multimodal.fusion import MultimodalConformalPredictor
from .online.tracker import OnlineConformalTracker
from .risk.control import ConformalRiskController
