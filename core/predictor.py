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
    Current Active: med_edge_tuned (v7.3.1 Robust)
    """
    return med_edge_tuned(a, b, c)

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
    Edge-Tuned MED v7.3.1 — Robustness fix for extreme neighbor jumps.
    Replaced branchless logic with explicit blocks to ensure Numba sign-safety.
    """
    max_ab: int = int(max(a, b))
    min_ab: int = int(min(a, b))
    diff: int = max_ab - min_ab
    ci: int = int(c)
    
    # Standard MED baseline
    p: int = min(max_ab, max(min_ab, int(a) + int(b) - ci))
    
    # Edge-Tuning: 
    # In smooth regions (diff 1-3), if C is an extreme outlier (>50 away),
    # we flip standard selection to track the neighbor closer to the step.
    if 1 <= diff <= 3:
        if ci > max_ab + 50:
            return np.uint8(max_ab)
        if ci < min_ab - 50:
            return np.uint8(min_ab)
            
    return np.uint8(p)


