"""
ZPNG-CSDE v6.2 [Flexible-Shard Architecture]
Module: zpng_predictor
Role: Pillar 2 - Prediction Kernels.
Description: Core spatial prediction algorithms (MED) and residual mapping (ZigZag).
Architecture: Pure Numba-JIT accelerated kernels for low-level spatial restoration.
"""

import numpy as np
from numba import njit

@njit(error_model='numpy', inline='always', cache=True)
def to_zigzag(val: int) -> np.uint8:
    """ Maps 8-bit signed residual to uint8 [Standard ZigZag]. """
    # Safe wrapping to 8-bit signed range [-128, 127]
    v_wrapped: int = (int(val) + 128) & 0xFF
    s: np.int8 = np.int8(v_wrapped - 128)
    return np.uint8((s << 1) ^ (s >> 7))

@njit(error_model='numpy', inline='always', cache=True)
def from_zigzag(z: np.uint8) -> int:
    """ Reverses ZigZag mapping back to signed integer. """
    return int(np.int8(z >> 1) ^ -(np.int8(z & 1)))

@njit(error_model='numpy', inline='always', cache=True)
def predict_med_standard(a: np.uint8, b: np.uint8, c: np.uint8) -> np.uint8:
    """
    Standard Median Edge Detector (MED).
    A: Left, B: Up, C: Top-Left (Up-Left).
    """
    mx: np.uint8 = max(a, b)
    mn: np.uint8 = min(a, b)
    gap: int = int(a) + int(b) - int(c)
    p: int = min(int(mx), max(int(mn), gap))
    return np.uint8(p)


@njit(error_model='numpy', inline='always', cache=True)
def predict_gap(a: np.uint8, b: np.uint8, c: np.uint8) -> np.uint8:
    """
    Gradient Adaptive Predictor (Simplified GAP).
    Dynamically switches between horizontal, vertical, or MED based on local gradient.
    """
    dv = abs(int(a) - int(c)) # horizontal gradient
    dh = abs(int(b) - int(c)) # vertical gradient
    if dv - dh > 32:
        return a
    if dh - dv > 32:
        return b
    return predict_med_standard(a, b, c)

@njit(error_model='numpy', inline='always', cache=True)
def predict_gradient(a: np.uint8, b: np.uint8, c: np.uint8) -> np.uint8:
    """ Average of A and B. Good for smooth isotropic areas. """
    return np.uint8((int(a) + int(b)) >> 1)

@njit(error_model='numpy', inline='always', cache=True)
def predict_left(a: np.uint8, b: np.uint8, c: np.uint8) -> np.uint8:
    """ Pure horizontal predictor. """
    return a

@njit(error_model='numpy', inline='always', cache=True)
def predict_dispatch(predictor_id: int, a: np.uint8, b: np.uint8, c: np.uint8) -> np.uint8:
    """ O(1) Dynamic routing for predictors. """
    if predictor_id == 0:
        return predict_med_standard(a, b, c)
    elif predictor_id == 1:
        return predict_gap(a, b, c)
    elif predictor_id == 2:
        return predict_gradient(a, b, c)
    else:
        return predict_left(a, b, c)

@njit(error_model='numpy', inline='always', cache=True)
def predict_paeth(a: np.uint8, b: np.uint8, c: np.uint8) -> np.uint8:
    # Phase 1 dummy fallback to prevent import errors before Phase 2 refactor
    return predict_med_standard(a, b, c)
