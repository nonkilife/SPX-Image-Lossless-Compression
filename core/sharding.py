"""
SPX [Stateless Sharding Hub]
Module: sharding
Role: Pillar 3 - Strategic Partitioning.
Description: Definitions for shard profiles and O(1) context derivation without global state.
Architecture: Profile-owned LUTs for zero-recompilation context derivation.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, uint8
from typing import List

from .shard_profile import (
    ShardProfile, PROFILE_RGB, get_shard_labels
)

SHARD_LABELS: List[str] = get_shard_labels(PROFILE_RGB)

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
