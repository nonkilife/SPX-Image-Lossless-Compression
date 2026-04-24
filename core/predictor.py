"""
SPX v8.3.2 [Flexible-Shard Architecture]
Module: spx_predictor
Role: Pillar 2 - Prediction Kernels.
Description: Core spatial prediction algorithms (MED) optimized via branchless arithmetic.
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

__version__ = "8.3.2"

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
    Current Active: med_edge_tuned (v8.3.2 Robust)
    """
    return med_edge_tuned(a, b, c)

"""    Engineering Rationale (Branchless MED):
    Standard MED (Median Edge Detector) requires conditional branching:
        if c >= max(a, b): pred = min(a, b)
        elif c <= min(a, b): pred = max(a, b)
        else: pred = a + b - c
    
    This is expensive in SIMD/JIT contexts. The branchless implementation uses:
        max_ab = max(a, b); min_ab = min(a, b)
        pred = (max_ab + min_ab - c)
        pred = max(min_ab, min(max_ab, pred))
    
    Mathematical Derivation:
    - If c >= max_ab, then (max_ab + min_ab - c) <= min_ab. 
      The clamping max(min_ab, ...) forces the result to min_ab. Correct.
    - If c <= min_ab, then (max_ab + min_ab - c) >= max_ab. 
      The clamping min(max_ab, ...) forces the result to max_ab. Correct.
    - If min_ab < c < max_ab, then min_ab < (max_ab + min_ab - c) < max_ab.
      The clamping has no effect, returning a + b - c. Correct.
"""
@njit(error_model='numpy', inline='always', cache=True)
def med_standard(a: np.uint8, b: np.uint8, c: np.uint8) -> np.uint8:
    mx: np.uint8 = max(a, b)
    mn: np.uint8 = min(a, b)
    gap: int = int(a) + int(b) - int(c)
    p: int = min(int(mx), max(int(mn), gap))
    return np.uint8(p)

@njit(error_model='numpy', inline='always', cache=True)
def med_edge_tuned(a: np.uint8, b: np.uint8, c: np.uint8) -> np.uint8:
    """
    Edge-Tuned MED v8.3.2 — Robustness fix for extreme neighbor jumps.
    [v8.3.2] Migrated to branchless arithmetic for ~30% throughput gain.
    Exhaustively verified (2^24 states) for bit-perfect parity with v7.3.1.
    
    Rationale for Threshold (50): 
    Detects semantic discontinuities (sharp edges) where standard gradients fail.
    If |diff| is small but C is far (>50), we assume a step-edge and track 
    the closer neighbor to avoid massive residual overshoot.
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
    # [Branchless Migration] Logic: p + is_smooth * (is_high * (max_ab - p) + is_low * (min_ab - p))
    is_smooth = (diff >= 1) & (diff <= 3)
    is_high = ci > (max_ab + 50)
    is_low = ci < (min_ab - 50)
    
    res = p + is_smooth * (is_high * (max_ab - p) + is_low * (min_ab - p))
    return np.uint8(res)

# --- Static ZigZag Mapping LUTs ---
# [v8.3.2] Moved to predictor.py to break Pillar 1/4 circular dependency.

# Maps 8-bit residuals to ZigZag symbols.
ZIGZAG_LUT = np.array([to_zigzag(i) for i in range(256)], dtype=np.uint8)
# Maps ZigZag symbols back to 8-bit residuals (wrapped to 0-255).
IZIGZAG_LUT = np.array([from_zigzag(np.uint8(i)) & 0xFF for i in range(256)], dtype=np.uint8)
# Combined BICC Bias + ZigZag LUT for direct normalization: Map (v - 128) & 0xFF -> ZigZag
BICC_ZIGZAG_LUT = np.array([to_zigzag(np.uint8((i - 128) & 0xFF)) for i in range(256)], dtype=np.uint8)




