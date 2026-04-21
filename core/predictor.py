"""
SPX v6.2 [Flexible-Shard Architecture]
Module: spx_predictor
Role: Pillar 2 - Prediction Kernels.
Description: Core spatial prediction algorithms (MED) and residual mapping (ZigZag).
Architecture: Pure Numba-JIT accelerated kernels for low-level spatial restoration.

Technical Flowchart:
```mermaid
graph TD
    Neighbors[Neighbors: A, B, C] --> MED{MED Logic}
    MED -->|Standard| SP[Standard Prediction]
    MED -->|Edge-Tuned| ET[Edge-Tuned Prediction]
    
    Val[Actual Pixel Value] --> Diff[Residual = Value - Prediction]
    Diff --> ZZ[ZigZag Mapping]
    ZZ --> Symbol[8-bit Entropy Symbol]
    
    Symbol --> IZZ[Inverse ZigZag]
    IZZ --> IDiff[Restore Residual]
    IDiff --> Rec[Prediction + Residual]
    Rec --> Final[Reconstructed Pixel]
```
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
def selected_predictor(a: np.uint8, b: np.uint8, c: np.uint8) -> np.uint8:
    """ 
    Unified Predictor Dispatcher. 
    Current Active: med_standard (v6.2 Stable)
    """
    return med_standard(a, b, c)

@njit(error_model='numpy', inline='always', cache=True)
def med_standard(a: np.uint8, b: np.uint8, c: np.uint8) -> np.uint8:
    """ Standard Median Edge Detector (MED). """
    mx: np.uint8 = max(a, b)
    mn: np.uint8 = min(a, b)
    gap: int = int(a) + int(b) - int(c)
    p: int = min(int(mx), max(int(mn), gap))
    return np.uint8(p)

@njit(error_model='numpy', inline='always', cache=True)
def med_edge_tuned(a: np.uint8, b: np.uint8, c: np.uint8) -> np.uint8:
    """
    Edge-Tuned MED v7.3.0 — branchless correction for extreme neighbor jumps.
    When max(a,b)-min(a,b) is 1-3 (smooth edge) but c deviates by >50 (step edge),
    the standard MED output is nudged by ±diff to track the dominant neighbor.
    """
    max_ab: int = int(max(a, b))
    min_ab: int = int(min(a, b))
    diff: int = max_ab - min_ab
    ci: int = int(c)
    p: int = min(max_ab, max(min_ab, int(a) + int(b) - ci))
    in_range: int = int(diff >= 1) * int(diff <= 3)
    flip1: int = int(ci - max_ab > 50) * in_range
    flip2: int = int(min_ab - ci > 50) * in_range
    return np.uint8((p + diff * (flip1 - flip2)) & 0xFF)


