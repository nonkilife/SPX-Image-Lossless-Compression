"""
SPX [Shard Profile Repository]
Module: shard_profile
Role: Pillar 3.1 - Strategic Configuration.
Description: Static data definitions for shard partitioning profiles.

Design Philosophy: Configuration-as-Data
----------------------------------------
This module isolates the mathematical design of shards (boundaries, mapping matrices) 
from the execution logic. By centralizing all profiles here, we enable rapid 
switching between different segmentation strategies (e.g., Universal vs. specialized 
High-Noise) without modifying the core context derivation kernels.
"""

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass(frozen=True)
class ShardProfile:
    """ 
    Authoritative physical architecture defining how context boundaries segment statistical space.
    Now encapsulates its own precomputed LUTs for stateless dispatch.
    """
    name: str
    v_boundaries_gr: npt.NDArray[np.uint8]
    intensity_segments: npt.NDArray[np.uint8]
    noise_shard_id: int  # -1 if no noise shard
    total_shards: int
    shard_map: npt.NDArray[np.uint8] # [v_level][intensity_idx][trend_idx]
    
    # Precomputed Dispatch LUTs
    spatial_lut: npt.NDArray[np.uint8]   # [511, 511]
    intensity_lut: npt.NDArray[np.uint8] # [256]
    dispatch_lut: npt.NDArray[np.uint8]  # [256, 4] 2D for fast JIT dispatch

def precompute_luts(v_bounds: npt.NDArray[np.uint8], 
                    i_segs: npt.NDArray[np.uint8], 
                    shard_map: npt.NDArray[np.uint8], 
                    nsid: int) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """
    Generates profile-specific LUTs for context features.
    """
    # 1. Intensity LUT
    i_lut = np.zeros(256, dtype=np.uint8)
    i_arr = np.arange(256, dtype=np.uint8)
    for thr in i_segs[1:-1]:
        i_lut += (i_arr > int(thr)).astype(np.uint8)

    # 2. Spatial Transition LUT (511x511, vectorized)
    s_lut = np.zeros((511, 511), dtype=np.uint8)
    d = np.arange(-255, 256, dtype=np.int16)
    DA, DB = np.meshgrid(d, d, indexing='ij')

    # Strength (V)
    v_mag = np.maximum(np.abs(DA), np.abs(DB))
    v_tier = (v_mag > 0).astype(np.uint8)
    for i in range(1, len(v_bounds) - 1):
        v_tier += (v_mag > int(v_bounds[i])).astype(np.uint8)

    # Trend (T)
    rising  = ((DA > 0) & (DB > 0)).astype(np.uint8)
    falling = ((DA < 0) & (DB < 0)).astype(np.uint8)
    t_idx   = (falling + 2 * (1 - rising - falling)).astype(np.uint8)

    # Noise Flag (N)
    ns_hit = ((np.abs(DA) > 12) & (np.abs(DB) > 12)).astype(np.uint8)

    # Packing: [V:3][T:2][N:1]
    s_lut[:] = ((v_tier << 3) | (t_idx << 1) | ns_hit).astype(np.uint8)

    # 3. Final Dispatch LUT (2D for branchless JIT access)
    d_lut = np.zeros((256, 4), dtype=np.uint8)
    n_v = shard_map.shape[0]
    n_t = shard_map.shape[2]
    
    for pk in range(256):
        vt = pk >> 3
        ti = (pk >> 1) & 0x03
        ns = pk & 0x01
        
        for ii in range(3):
            if ns != 0 and nsid >= 0:
                d_lut[pk, ii] = np.uint8(nsid)
            elif vt < n_v and ti < n_t:
                # shard_map is [v, i, t]
                d_lut[pk, ii] = shard_map[vt, ii, ti]
            else:
                # Fallback to noise shard or last valid shard
                d_lut[pk, ii] = np.uint8(nsid if nsid >= 0 else 0)
    
    return s_lut, i_lut, d_lut

# --- 1. Universal-42 Profile (Default) ---

V_BOUND_RGB = np.array([0, 1, 2, 4, 8, 16, 32, 255], dtype=np.uint8)
INTENSITY_SEG_RGB = np.array([0, 60, 190, 255], dtype=np.uint8)

def build_shard_map_universal_42() -> npt.NDArray[np.uint8]:
    """ Unified 42-shard balanced architecture: 3I×1T flat | 3I×3T full | 1I×3T trend-only. """
    s_map = np.zeros((8, 3, 3), dtype=np.uint8)
    
    # Tier 0 (V=0): Intensity Split (IDs 0-2)
    # Rationale: For flat regions, only brightness determines entropy.
    for i in range(3): 
        s_map[0, i, :] = i
    
    # Tier 1, 2, 3 (V=1, 2, 3): Intensity * Trend (IDs 3-29)
    # 3 tiers * 9 = 27 shards
    # Rationale: For low-to-medium gradients, both direction and brightness are informative.
    for v in range(1, 4):
        for i in range(3):
            base = 3 + (v-1) * 9 + i * 3
            s_map[v, i, 0] = base + 0
            s_map[v, i, 1] = base + 1
            s_map[v, i, 2] = base + 2
            
    # Tiers 4, 5, 6, 7 (V >= 4): Trend-only (IDs 30-41)
    # 4 tiers * 3 trends = 12 shards
    # Rationale: For strong edges, the exact brightness of the pixel is less predictive 
    # than the directional trend of the gradient.
    for v in range(4, 8):
        for i in range(3):
            base = 30 + (v-4) * 3
            s_map[v, i, 0] = base + 0
            s_map[v, i, 1] = base + 1
            s_map[v, i, 2] = base + 2
            
    return s_map

_s_lut_rgb, _i_lut_rgb, _d_lut_rgb = precompute_luts(V_BOUND_RGB, INTENSITY_SEG_RGB, build_shard_map_universal_42(), -1)

PROFILE_RGB = ShardProfile(
    name="Universal-42",
    v_boundaries_gr=V_BOUND_RGB,
    intensity_segments=INTENSITY_SEG_RGB,
    noise_shard_id=-1,
    total_shards=42,
    shard_map=build_shard_map_universal_42(),
    spatial_lut=_s_lut_rgb,
    intensity_lut=_i_lut_rgb,
    dispatch_lut=_d_lut_rgb
)

def get_shard_labels(profile: ShardProfile) -> List[str]:
    """ Generates generic index labels for all shards in the given profile. """
    return [f"Shard_{i}" for i in range(profile.total_shards)]
