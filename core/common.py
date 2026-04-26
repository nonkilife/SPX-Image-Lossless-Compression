"""
SPX v8.3.2 [Foundational Protocol]
Module: common
Role: Pillar 1 - Core Protocol & Constants.
Description: Centralized definitions for mode flags, coding thresholds, and profile defaults.
Architecture: Pillar-based modular codec with experience-based rANS probability templates.

Logic Path (Context ID Derivation):
```mermaid
graph TD
    In[Input: ag, bg, cg] --> ST[Double Feature Lookup: spatial_lut]
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

__version__ = "8.3.2"

# [v8.3.2] Lazy Initialization for Empirical Templates to reduce Import Overhead
# These templates represent the "statistical DNA" of natural images.
_CACHED_TEMPLATES: Optional[npt.NDArray[np.uint64]] = None

def get_empirical_templates() -> npt.NDArray[np.uint64]:
    """ 
    Provides the 30 experience-based PDF templates for the sharding engine.
    Loaded from 'rans_mode.npz' and cached for performance.
    
    Technical Matrix:
    - 10 Hybrid Elite V10 centroids (categorical shapes + K-Means centroids).
    - 3 Scaling factors [0.5, 1.0, 1.5] for dynamic range adaptation.
    - Total 30 modes. Mode_ID = (scale_idx * 10) + centroid_idx.
    
    Why Templates?
    Instead of saving a full 256-symbol frequency table for every shard (which would 
    destroy compression ratio for small shards), we select a pre-computed "Mode" 
    that best fits the data. This reduces the header overhead from ~500 bytes per 
    shard to just 1 byte.
    """
    global _CACHED_TEMPLATES
    if _CACHED_TEMPLATES is None:
        # Resolve absolute path to the sibling .npz file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "rans_mode.npz")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"SPX Critical Error: Missing rANS templates at {path}")
            
        try:
            with np.load(path) as data:
                if 'templates' not in data:
                    raise KeyError("Key 'templates' not found in rans_mode.npz")
                _CACHED_TEMPLATES = data['templates']
        except Exception as e:
            raise RuntimeError(f"SPX Critical Error: Failed to load rANS templates: {e}")
            
    return _CACHED_TEMPLATES

# --- 1. Protocol Constants (Header Flags) ---
# [v8.3.2] Standardized Bitstream Protocol Flags.
# These flags define the fundamental processing path for the entire image.

# Image has an Alpha channel. Alpha is currently compressed using Zstd Level 1.
FLAG_RGBA: int         = 0x01

# Optimized monochrome engine. Prunes R/B channels to save 66% compute.
# Does NOT use RCT; operates directly on the Green/Luma channel.
FLAG_GRAYSCALE: int    = 0x10

# Adaptive Green-Subtract Transform (Reversible Color Transform).
# Essential for RGB photographic images to decorrelate color channels.
# Path: Grn = G; RedDiff = R - G; BluDiff = B - G.
FLAG_COLOR_GSUB: int   = 0x20

# 2D Bit-Context engine (Bitplane rANS).
# Used for high-entropy images where standard context-sharding hits a plateau.
# Decomposes residuals into 2-bit layers with spatial context modeling.
FLAG_BITPLANE: int     = 0x40

# --- 2. Bitplane rANS Sensitivity Thresholds ---
# [v8.3.2] Calibrated for Zero-Regression on Photographic Datasets.
# 
# Engineering Rationale:
# - BITPLANE_H_THRESHOLD (3.2): Threshold where the overhead of 2-bit spatial context 
#   derivation usually offsets the coding gains for high-entropy noisy regions.
# - BITPLANE_HIT_RATE_THRESHOLD (0.30): Minimum required fraction of zero-residuals 
#   to ensure the bitplane's sparse coding model is effective.
# - P90 < 112 Rationale: Ensures the residual width is strictly controlled to 
#   prevents bitplane triggering on complex photographic noise where standard rANS excels.

BITPLANE_H_THRESHOLD: float = 3.2          # Shannon Entropy Gating (bits/symbol)
BITPLANE_HIT_RATE_THRESHOLD: float = 0.30    # Minimum Zero-Residual Fraction
BITPLANE_P90_THRESHOLD: int = 112             # Max 90th-percentile ZigZag symbol width

