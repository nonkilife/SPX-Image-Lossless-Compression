"""
SPX [Stateless Sharding Hub]
Module: sharding
Role: Pillar 3 - Strategic Partitioning.
Description: Defintions for shard profiles and O(1) context derivation without global state.
Architecture: Profile-owned LUTs for zero-recompilation context derivation.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, uint8, uint16
from typing import Tuple, Optional, List
from dataclasses import dataclass

# --- 1. Sharding Profile System ---

@dataclass(frozen=True)
class ShardProfile:
    """ 
    Authoritative physical architecture defining how context boundaries segment statistical space.
    Now encapsulates its own precomputed LUTs for stateless dispatch.
    """
    name: str
    total_shards: int
    v_boundaries_gr: npt.NDArray[np.uint8]
    intensity_segments: npt.NDArray[np.uint8]
    noise_shard_id: int
    shard_map: npt.NDArray[np.uint8] # [v_level][intensity_idx][trend_idx]
    
    # Precomputed Dispatch LUTs
    spatial_lut: npt.NDArray[np.uint8]   # [511, 511]
    intensity_lut: npt.NDArray[np.uint8] # [256]
    dispatch_lut: npt.NDArray[np.uint8]  # [1024]

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

    # 3. Final Dispatch LUT
    d_lut = np.zeros(1024, dtype=np.uint8)
    n_v = shard_map.shape[0]
    n_t = shard_map.shape[2]
    for pk in range(256):
        vt = pk >> 3
        ti = (pk >> 1) & 0x03
        ns = pk & 0x01
        for ii in range(3):
            idx = (pk << 2) | ii
            if ns != 0 and nsid >= 0:
                d_lut[idx] = np.uint8(nsid)
            elif vt < n_v and ti < n_t:
                d_lut[idx] = shard_map[vt, ii, ti]
    
    return s_lut, i_lut, d_lut

# --- 2. Default Profile: Universal-42 ---
V_BOUND_RGB = np.array([0, 1, 2, 4, 8, 16, 32, 255], dtype=np.uint8)
INTENSITY_SEG_RGB = np.array([0, 60, 190, 255], dtype=np.uint8)

def build_shard_map_universal_42() -> npt.NDArray[np.uint8]:
    s_map = np.zeros((8, 3, 3), dtype=np.uint8)
    for i in range(3): s_map[0, i, :] = i
    for v in range(1, 4):
        for i in range(3):
            base = 3 + (v-1) * 9 + i * 3
            s_map[v, i, 0] = base + 0
            s_map[v, i, 1] = base + 1
            s_map[v, i, 2] = base + 2
    for v in range(4, 8):
        for i in range(3):
            base = 30 + (v-4) * 3
            s_map[v, i, 0] = base + 0
            s_map[v, i, 1] = base + 1
            s_map[v, i, 2] = base + 2
    return s_map

# Global profile instantiation with LUT precomputation
_s_lut_rgb, _i_lut_rgb, _d_lut_rgb = precompute_luts(V_BOUND_RGB, INTENSITY_SEG_RGB, build_shard_map_universal_42(), -1)

PROFILE_RGB = ShardProfile(
    name="Universal-42",
    total_shards=42,
    v_boundaries_gr=V_BOUND_RGB,
    intensity_segments=INTENSITY_SEG_RGB,
    noise_shard_id=-1,
    shard_map=build_shard_map_universal_42(),
    spatial_lut=_s_lut_rgb,
    intensity_lut=_i_lut_rgb,
    dispatch_lut=_d_lut_rgb
)

def get_shard_labels(n_shards: Optional[int] = None) -> List[str]:
    if n_shards is None:
        n_shards = PROFILE_RGB.total_shards
    return [f"Shard_{i}" for i in range(n_shards)]

SHARD_LABELS: List[str] = get_shard_labels()

@njit(inline='always', fastmath=True)
def get_context_id_fast(ag: uint8, bg: uint8, cg: uint8, intensity: uint8,
                        s_lut: npt.NDArray[np.uint8], 
                        i_lut: npt.NDArray[np.uint8], 
                        d_lut: npt.NDArray[np.uint8]) -> uint8:
    """
    Stateless Triple LUT dispatch.
    """
    # Cast to int for indexing safety in Numba
    packed = s_lut[int(ag) - int(cg) + 255, int(bg) - int(cg) + 255]
    i_idx  = i_lut[int(intensity)]
    return d_lut[(int(packed) << 2) | int(i_idx)]
