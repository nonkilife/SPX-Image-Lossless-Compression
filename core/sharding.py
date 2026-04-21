"""
SPX [Unified Sharding Hub]
Module: sharding
Role: Pillar 3 - Strategic Partitioning.
Description: Authoritative definitions for shard profiles, mapping matrices, and context derivation.
Architecture: Flexible Sharding Hub utilizing 3D Mapping LUTs for zero-latency context ID derivation.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, uint8
from typing import Tuple, Optional, List
from dataclasses import dataclass

# --- 1. Sharding Profile System ---

@dataclass(frozen=True)
class ShardProfile:
    """ 
    Authoritative physical architecture defining how context boundaries segment statistical space.
    
    The Shard Map establishes a rigid mathematical space utilizing intensity limits, 
    gradient variance tiers, and local curve slopes to reliably bucket pixels with exactly 
    matching neighborhood configurations into the same dynamic compression container.
    """
    name: str
    v_boundaries_gr: npt.NDArray[np.uint8]
    intensity_segments: npt.NDArray[np.uint8]
    noise_shard_id: int  # -1 if no noise shard
    total_shards: int
    shard_map: npt.NDArray[np.uint8] # [v_level][intensity_idx][trend_idx]

# --- 2. Default Profile: Universal-42 ---
V_BOUND_RGB = np.array([0, 1, 2, 4, 8, 16, 32, 255], dtype=np.uint8)
INTENSITY_SEG_RGB = np.array([0, 60, 190, 255], dtype=np.uint8)

def build_shard_map_universal_42() -> npt.NDArray[np.uint8]:
    """ Unified 42-shard balanced architecture: 3I×1T flat | 3I×3T full | 1I×3T trend-only. """
    s_map = np.zeros((8, 3, 3), dtype=np.uint8)
    # Tier 0 (V=0): Intensity Split (IDs 0-2)
    for i in range(3): s_map[0, i, :] = i
    
    # Tier 1, 2, 3 (V=1, 2, 3): Intensity * Trend (IDs 3-29)
    # 3 tiers * 9 = 27 shards
    for v in range(1, 4):
        for i in range(3):
            base = 3 + (v-1) * 9 + i * 3
            s_map[v, i, 0] = base + 0
            s_map[v, i, 1] = base + 1
            s_map[v, i, 2] = base + 2
            
    # Tiers 4, 5, 6, 7 (V >= 4): Trend-only (IDs 30-41)
    # 4 tiers * 3 trends = 12 shards
    for v in range(4, 8):
        for i in range(3):
            base = 30 + (v-4) * 3
            s_map[v, i, 0] = base + 0
            s_map[v, i, 1] = base + 1
            s_map[v, i, 2] = base + 2
            
    return s_map
    
PROFILE_RGB = ShardProfile(
    name="Universal-42",
    v_boundaries_gr=V_BOUND_RGB,
    intensity_segments=INTENSITY_SEG_RGB,
    noise_shard_id=-1,
    total_shards=42,
    shard_map=build_shard_map_universal_42()
)

def get_shard_labels(n_shards: Optional[int] = None) -> List[str]:
    """ Generates generic index labels for all shards in the given (or current) profile. """
    if n_shards is None:
        n_shards = PROFILE_RGB.total_shards
    return [f"Shard_{i}" for i in range(n_shards)]

# --- 3. High-Performance Context Dispatcher LUTs ---
SPATIAL_TRANS_LUT = np.zeros((511, 511), dtype=np.uint8)
INTENSITY_LUT = np.zeros(256, dtype=np.uint8)
FINAL_DISPATCH_LUT = np.zeros(1024, dtype=np.uint8)  # index = (packed<<2)|i_idx

def initialize_luts_python(v_bounds, i_segs, shard_map, nsid: int):
    """
    Fills global LUTs with precomputed context features.
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
    """
    global _LAST_V_BOUNDS, _LAST_I_SEGS, _LAST_NSID
    if (not np.array_equal(_LAST_V_BOUNDS, v_bounds) or
            not np.array_equal(_LAST_I_SEGS, i_segs) or
            _LAST_NSID != nsid):
        initialize_luts_python(v_bounds, i_segs, shard_map, nsid)
        _LAST_V_BOUNDS = v_bounds.copy()
        _LAST_I_SEGS = i_segs.copy()
        _LAST_NSID = nsid

# Initial load
sync_luts_if_needed(V_BOUND_RGB, INTENSITY_SEG_RGB, PROFILE_RGB.shard_map, PROFILE_RGB.noise_shard_id)

SHARD_LABELS: List[str] = get_shard_labels()

@njit(inline='always', fastmath=True, cache=True)
def get_context_id_fast(ag: uint8, bg: uint8, cg: uint8, intensity: uint8,
                        shard_map: npt.NDArray[np.uint8], nsid: int) -> uint8:
    """
    Triple LUT dispatch: SPATIAL_TRANS_LUT -> INTENSITY_LUT -> FINAL_DISPATCH_LUT.
    """
    packed = SPATIAL_TRANS_LUT[int(ag) - int(cg) + 255, int(bg) - int(cg) + 255]
    i_idx  = INTENSITY_LUT[intensity]
    return FINAL_DISPATCH_LUT[(packed << 2) | i_idx]
