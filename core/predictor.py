"""
ZPNG-CSDE v6.2 [Flexible-Shard Architecture]
Module: zpng_predictor
Role: Pillar 2 - Prediction Kernels.
Description: Core spatial prediction algorithms (MED) and residual mapping (ZigZag).
Architecture: Pure Numba-JIT accelerated kernels for low-level spatial restoration.
"""

import numpy as np
from numba import njit

@njit(error_model='numpy', cache=True)
def to_zigzag(val: int) -> np.uint8:
    """ Maps 8-bit signed residual to uint8 [Standard ZigZag]. """
    # Safe wrapping to 8-bit signed range [-128, 127]
    v_wrapped: int = (int(val) + 128) & 0xFF
    s: np.int8 = np.int8(v_wrapped - 128)
    return np.uint8((s << 1) ^ (s >> 7))

@njit(fastmath=True, error_model='numpy', inline='always', cache=True)
def from_zigzag(z: np.uint8) -> int:
    """ Reverses ZigZag mapping back to signed integer. """
    return int(np.int8(z >> 1) ^ -(np.int8(z & 1)))

@njit(fastmath=True, error_model='numpy', inline='always', cache=True)
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
