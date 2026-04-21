"""
SPX [Unified Sharding Coordination Hub]
Module: sharding
Role: Pillar 3 - Statistical Partitioning & Dispatch.
Description: The authoritative execution engine for pixel sharding. 

Design Philosophy: Logic-Data Separation
----------------------------------------
This module acts as the "Coordinator" that bridges static configurations with 
high-performance execution. It manages the lifecycle of the Global Context LUTs 
(Look-Up Tables) and provides the JIT-compiled kernels used by the parallel 
image processing pipelines.

Architecture:
1. Data Hub: Imports and re-exports ShardProfiles (from shard_profile.py).
2. Synchronization: Monitors profile changes and updates global LUTs (V/I/T).
3. Context Engine: Triple-LUT dispatch for O(1) context ID derivation.

Logic Path:
```mermaid
graph TD
    A[ShardProfile] -->|sync_luts_if_needed| B{State Cache}
    B -->|Change Detected| C[initialize_luts_python]
    B -->|No Change| D[Use Existing LUTs]
    C --> E[Update Global LUTs: SPATIAL/INTENSITY/FINAL]
    
    subgraph Execution Kernel
    F[Neighbors & Intensity] --> G[get_context_id_fast]
    G --> H[LUT 1: Spatial Packaging]
    H --> I[LUT 2: Intensity Logic]
    I --> J[LUT 3: Final Dispatch]
    J --> K[Context ID]
    end
    
    E -.-> G
    D -.-> G
```
"""

import numpy as np
import numpy.typing as npt
from numba import njit, uint8
from typing import Tuple, Optional, List

# --- 1. Authorized Re-exports ---
# These are pulled from shard_profile.py to ensure this module remains the 
# single point of contact for all sharding operations.
from .shard_profile import (
    ShardProfile, PROFILE_RGB, get_shard_labels
)

# --- 1. High-Performance Context Dispatcher LUTs ---
# These global tables are shared across all JIT kernels for zero-copy feature extraction.
SPATIAL_TRANS_LUT = np.zeros((511, 511), dtype=np.uint8)
INTENSITY_LUT = np.zeros(256, dtype=np.uint8)
FINAL_DISPATCH_LUT = np.zeros(1024, dtype=np.uint8)  # index = (packed<<2)|i_idx

def initialize_luts_python(v_bounds, i_segs, shard_map, nsid: int):
    """
    Fills global LUTs with precomputed context features based on profile boundaries.
    """
    # 1. Intensity LUT
    i_arr = np.arange(256, dtype=np.uint8)
    i_result = np.zeros(256, dtype=np.uint8)
    for thr in i_segs[1:-1]:
        i_result += (i_arr > int(thr)).astype(np.uint8)
    INTENSITY_LUT[:] = i_result

    # 2. Spatial Transition LUT (511x511, vectorized)
    d = np.arange(-255, 256, dtype=np.int16)
    DA, DB = np.meshgrid(d, d, indexing='ij')

    # Strength (V)
    v = np.maximum(np.abs(DA), np.abs(DB))
    v_tier = (v > 0).astype(np.uint8)
    for i in range(1, len(v_bounds) - 1):
        v_tier += (v > int(v_bounds[i])).astype(np.uint8)

    # Trend (T)
    rising  = ((DA > 0) & (DB > 0)).astype(np.uint8)
    falling = ((DA < 0) & (DB < 0)).astype(np.uint8)
    t_idx   = (falling + 2 * (1 - rising - falling)).astype(np.uint8)

    # Noise Flag (N)
    ns_hit = ((np.abs(DA) > 12) & (np.abs(DB) > 12)).astype(np.uint8)

    # Packing: [V:3][T:2][N:1]
    SPATIAL_TRANS_LUT[:] = ((v_tier << 3) | (t_idx << 1) | ns_hit).astype(np.uint8)

    # 3. Final Dispatch LUT
    n_v = shard_map.shape[0]
    n_t = shard_map.shape[2]
    for pk in range(256):
        vt = pk >> 3
        ti = (pk >> 1) & 0x03
        ns = pk & 0x01
        for ii in range(3):
            idx = (pk << 2) | ii
            if ns != 0 and nsid >= 0:
                FINAL_DISPATCH_LUT[idx] = np.uint8(nsid)
            elif vt < n_v and ti < n_t:
                FINAL_DISPATCH_LUT[idx] = shard_map[vt, ii, ti]

_LAST_V_BOUNDS = np.zeros(8, dtype=np.uint8)
_LAST_I_SEGS = np.zeros(4, dtype=np.uint8)
_LAST_NSID: int = -2

def sync_luts_if_needed(v_bounds, i_segs, shard_map, nsid: int):
    """
    Ensures global LUTs match the requested profile.
    Only recalculates if boundaries have changed since the last invocation.
    """
    global _LAST_V_BOUNDS, _LAST_I_SEGS, _LAST_NSID
    if (not np.array_equal(_LAST_V_BOUNDS, v_bounds) or
            not np.array_equal(_LAST_I_SEGS, i_segs) or
            _LAST_NSID != nsid):
        initialize_luts_python(v_bounds, i_segs, shard_map, nsid)
        _LAST_V_BOUNDS = v_bounds.copy()
        _LAST_I_SEGS = i_segs.copy()
        _LAST_NSID = nsid

# Initial load with the default Universal-42 profile
sync_luts_if_needed(PROFILE_RGB.v_boundaries_gr, PROFILE_RGB.intensity_segments, PROFILE_RGB.shard_map, PROFILE_RGB.noise_shard_id)

SHARD_LABELS: List[str] = get_shard_labels(PROFILE_RGB)

@njit(inline='always', fastmath=True, cache=True)
def get_context_id_fast(ag: uint8, bg: uint8, cg: uint8, intensity: uint8,
                        shard_map: npt.NDArray[np.uint8], nsid: int) -> uint8:
    """
    High-speed context derivation kernel.
    Triple LUT dispatch: SPATIAL_TRANS_LUT -> INTENSITY_LUT -> FINAL_DISPATCH_LUT.
    """
    packed = SPATIAL_TRANS_LUT[int(ag) - int(cg) + 255, int(bg) - int(cg) + 255]
    i_idx  = INTENSITY_LUT[intensity]
    return FINAL_DISPATCH_LUT[(packed << 2) | i_idx]
