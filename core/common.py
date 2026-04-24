"""
SPX v8.3.2 [Foundational Protocol]
Module: common
Role: Pillar 1 - Core Protocol & Constants.
Description: Centralized definitions for mode flags, coding thresholds, and profile defaults.
Architecture: Pillar-based modular codec with experience-based rANS probability templates.

Logic Path (Context ID Derivation):
```mermaid
graph TD
    In[Input: ag, bg, cg, LeadValue] --> ST[Double Feature Lookup: spatial_lut]
    In --> IL[Intensity Lookup: intensity_lut]
    ST --> Feat[Extract: v_tier, t_idx, ns_hit]
    IL --> Feat
    Feat --> CID[Final Context ID via Dispatch LUT]
    %% Spatial lookup uses (ag-cg) and (bg-cg) to achieve local DC-invariance.
    %% Packing Format: [V_Tier:3 bits] | [Trend:2 bits] | [Noise:1 bit]
```
"""

import numpy as np
import numpy.typing as npt
import os
from typing import Optional

# [v8.3.2] Lazy Initialization for Empirical Templates to reduce Import Overhead
_CACHED_TEMPLATES: Optional[npt.NDArray[np.uint64]] = None

def get_empirical_templates() -> npt.NDArray[np.uint64]:
    """ 
    Provides the 30 experience-based PDF templates for the sharding engine.
    Loaded from 'rans_mode.npz' and cached for performance.
    
    Technical Matrix:
    - 10 Hybrid Elite V10 centroids (categorical shapes + K-Means centroids).
    - 3 Scaling factors [0.5, 1.0, 1.5] for dynamic range adaptation.
    - Total 30 modes. Mode_ID = (scale_idx * 10) + centroid_idx.
    """
    global _CACHED_TEMPLATES
    if _CACHED_TEMPLATES is None:
        # Resolve absolute path to the sibling .npz file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "rans_mode.npz")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"SPX Critical Error: Missing rANS templates at {path}")
            
        with np.load(path) as data:
            _CACHED_TEMPLATES = data['templates']
            
    return _CACHED_TEMPLATES

# --- 1. Protocol Constants (Header Flags) ---
# [v8.3.2] Standardized Bitstream Protocol Flags
FLAG_RGBA: int        = 0x01
FLAG_SIMPLE: int      = 0x02  # Zstd-compressed Raw Pixels (No sharding)
FLAG_RAW: int         = 0x04  # Uncompressed Raw Pixels (No zstd)
FLAG_PASSTHROUGH: int = 0x08  # Original File Storage (PNG/JPG)
FLAG_GRAYSCALE: int   = 0x10  # Hardware-accelerated true monochrome bypassing RCT
FLAG_COLOR_GSUB: int   = 0x20  # Adaptive Green-Subtract Transform (Smooth Image Optimization)
FLAG_BITPLANE: int    = 0x40  # 2D Bit-Context engine (BICC Stage 2)

# --- 2. Bitplane rANS Sensitivity Thresholds ---
# [v8.3.2] Calibrated for Zero-Regression on Photographic Datasets.
# 
# Engineering Rationale:
# - BITPLANE_H_THRESHOLD (3.2): Threshold where the overhead of 2-bit spatial context 
#   derivation usually offsets the coding gains for high-entropy noisy regions.
# - BITPLANE_HIT_RATE_THRESHOLD (0.30): Minimum required fraction of zero-residuals 
#   to ensure the bitplane's sparse coding model is effective.
# - P90 < 112 Rationale: The 90th percentile of residuals must be below 112 to 
#   ensure the "residual noise" doesn't overwhelm the spatial context derivation. 
#   This filter prevents the bitplane mode from triggering on complex natural textures 
#   where standard rANS is more efficient.
BITPLANE_H_THRESHOLD: float = 3.2          # Shannon Entropy Gating (bits/symbol)
BITPLANE_HIT_RATE_THRESHOLD: float = 0.30    # Minimum Zero-Residual Fraction
BITPLANE_P90_THRESHOLD: int = 112             # Max 90th-percentile ZigZag symbol width

ENABLE_DIAGNOSTICS: bool = False  # Production Gate
