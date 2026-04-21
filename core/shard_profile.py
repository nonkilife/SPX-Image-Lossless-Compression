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
from typing import List, Optional

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

PROFILE_RGB = ShardProfile(
    name="Universal-42",
    v_boundaries_gr=V_BOUND_RGB,
    intensity_segments=INTENSITY_SEG_RGB,
    noise_shard_id=-1,
    total_shards=42,
    shard_map=build_shard_map_universal_42()
)

def get_shard_labels(profile: ShardProfile) -> List[str]:
    """ Generates generic index labels for all shards in the given profile. """
    return [f"Shard_{i}" for i in range(profile.total_shards)]
