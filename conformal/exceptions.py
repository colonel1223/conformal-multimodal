"""Exception hierarchy for conformal-multimodal."""


class ConformalError(Exception):
    """Base exception for all conformal prediction errors."""
    pass


class NotCalibratedError(ConformalError):
    """Raised when predict() is called before calibrate()."""
    def __init__(self, method_name: str = "predictor"):
        super().__init__(
            f"{method_name} has not been calibrated. "
            f"Call calibrate() with held-out data before predict()."
        )


class InsufficientDataError(ConformalError):
    """Raised when calibration set is too small for meaningful guarantees."""
    def __init__(self, n: int, min_required: int):
        super().__init__(
            f"Calibration set has {n} points, but at least {min_required} "
            f"are needed for the requested coverage level. With {n} points, "
            f"the tightest achievable miscoverage is {1/(n+1):.4f}."
        )


class ModalityMismatchError(ConformalError):
    """Raised when modality keys don't match between calibration and prediction."""
    def __init__(self, expected: set, got: set):
        missing = expected - got
        extra = got - expected
        msg = "Modality mismatch."
        if missing:
            msg += f" Missing: {missing}."
        if extra:
            msg += f" Unexpected: {extra}."
        super().__init__(msg)
