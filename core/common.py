"""
SPX [Flexible Shard Architecture]
Module: common
Role: Pillar 1 - Foundation & Protocol.
Description: Authoritative definitions for constants, sharding matrices, and header flags.
Architecture: Flexible Sharding Hub utilizing 3D Mapping LUTs for context ID derivation.

Logic Path:
```mermaid
graph TD
    In[Input: ag, bg, cg, LeadValue] --> ST[Double Feature Lookup: SPATIAL_TRANS_LUT]
    In --> IL[Intensity Lookup: INTENSITY_LUT]
    ST --> Feat[Extract: v_tier, t_idx, ns_hit]
    IL --> Feat
    Feat --> ShMap[ShardProfile.shard_map]
    ShMap --> CID[Final Context ID]
    %% Spatial lookup uses ag-cg, bg-cg to achieve DC-invariance.
    %% Packing Format: [V_Tier:3 bits] | [Trend:2 bits] | [Noise:1 bit]
```
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange, uint8
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
from .predictor import to_zigzag, from_zigzag, selected_predictor, med_edge_tuned
from .sharding import ShardProfile, PROFILE_RGB, SHARD_LABELS, sync_luts_if_needed, get_context_id_fast

# --- 0. Empirical Model Pillars ---

def _build_empirical_templates() -> Tuple[npt.NDArray[np.uint64], ...]:
    """
    30-template matrix: 10 Hybrid Elite V10 centroids × 3 scales [0.5, 1.0, 1.5].
    7 categorical shapes + 3 universal K-Means centroids. Mode_ID = 4 + (scale_idx * 10) + centroid_idx.
    """
    P_LIST = [
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,3,4,4,5,6,8,10,13,17,22,30,42,59,83,116,160,216,280,345,398,433,393,338,274,211,157,114,82,59,43,31,23,18,14,11,8,7,6,5,4,3,3,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,3,3,4,5,7,10,14,21,33,55,92,156,257,395,545,659,583,446,300,185,110,65,39,25,17,12,9,6,5,4,3,3,2,2,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,2,4,6,11,23,48,104,217,421,716,992,709,414,215,104,49,24,12,7,4,3,2,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,2,4,7,14,31,75,180,399,766,1130,763,399,181,76,32,14,7,4,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,4,7,17,47,133,357,810,1329,812,357,135,48,18,8,4,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,3,8,23,78,276,828,1647,833,279,79,23,8,3,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,5,5,6,7,8,9,10,12,13,15,18,21,25,29,35,41,49,58,70,83,98,116,136,158,180,203,224,241,270,240,221,199,176,153,132,113,96,81,68,58,49,41,35,30,25,22,19,16,14,12,11,9,8,7,7,6,5,5,4,4,3,3,3,3,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,4,4,4,5,6,6,7,8,9,10,12,13,15,17,19,22,25,28,32,36,41,46,52,59,66,75,84,95,107,120,135,152,170,190,210,248,212,191,171,151,134,118,105,93,82,73,65,58,52,46,41,36,32,28,25,23,20,18,16,14,13,11,10,9,8,7,7,6,5,5,4,4,4,3,3,3,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,3,4,6,9,14,23,45,93,204,431,761,989,714,396,190,89,43,23,14,9,6,4,3,2,2,2,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,5,5,6,6,7,8,8,9,10,11,12,14,15,17,19,20,23,25,27,30,33,37,41,44,49,53,59,64,70,77,84,92,100,110,120,131,145,159,174,207,174,158,144,130,117,106,97,88,80,73,66,60,55,50,46,42,38,35,31,29,26,24,22,20,18,16,15,14,12,11,10,9,9,8,7,7,6,5,5,5,4,4,4,3,3,3,3,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
    ]
    SCALES = [0.5, 1.0, 1.5]
    final_templates = []
    for scale in SCALES:
        for base_p in P_LIST:
            new_p = np.zeros(256, dtype=np.float64)
            for i in range(256):
                src_i = 128 + (i - 128) / scale
                idx = int(round(src_i))
                if 0 <= idx < 256: new_p[i] = base_p[idx]
            new_p += 0.5
            total_mass = np.sum(new_p)
            scaled_v = (new_p * 4096.0 / total_mass).astype(np.uint64)
            diff = 4096 - np.sum(scaled_v)
            peak_idx = np.argmax(scaled_v)
            scaled_v[peak_idx] = np.uint64(int(scaled_v[peak_idx]) + int(diff))
            final_templates.append(scaled_v)
    return tuple(final_templates)

EMPIRICAL_TEMPLATES: Tuple[npt.NDArray[np.uint64], ...] = _build_empirical_templates()

# --- 1. Protocol Constants (Header Flags) ---
FLAG_RGBA: int        = 0x01
FLAG_SIMPLE: int      = 0x02  # Zstd-compressed Raw Pixels (No sharding)
FLAG_RAW: int         = 0x04  # Uncompressed Raw Pixels (No zstd)
FLAG_PASSTHROUGH: int = 0x08  # Original File Storage (PNG/JPG)
FLAG_GRAYSCALE: int   = 0x10  # Hardware-accelerated true monochrome bypassing CR
FLAG_COLOR_GSUB: int   = 0x20  # Adaptive Green-Subtract Transform (Smooth Image Optimization)
FLAG_BITPLANE: int    = 0x40  # 2D Bit-Context engine (BICC Stage 2)

# --- 2. Data Structures ---

@dataclass
class SpxResult:
    """ Unified container for SPX compression/decompression metrics. """
    # Timing (seconds)
    enc_time: float = 0.0
    dec_time: float = 0.0
    
    # Core Metadata
    h: int = 0
    w: int = 0
    is_rgba: bool = False
    comp_size: int = 0
    orig_size: int = 0
    
    # Statistical Diagnostics
    hits: npt.NDArray[np.uint32] = field(default_factory=lambda: np.zeros(4, dtype=np.uint32))
    res_sums: npt.NDArray[np.uint64] = field(default_factory=lambda: np.zeros(4, dtype=np.uint64))
    # Sharding (sized to active profile at instantiation time)
    shard_counts: npt.NDArray[np.uint32] = field(default_factory=lambda: np.zeros((3, PROFILE_RGB.total_shards), dtype=np.uint32))
    shard_ptrs: Optional[Tuple] = None
    shard_stats: npt.NDArray[np.uint32] = field(default_factory=lambda: np.zeros((3, PROFILE_RGB.total_shards, 256), dtype=np.uint32))
    shard_widths: npt.NDArray[np.uint16] = field(default_factory=lambda: np.zeros((3, PROFILE_RGB.total_shards), dtype=np.uint16))

    # Channel Statistical Data (Global Histograms: Grn, RD, BD)
    channel_hists: npt.NDArray[np.uint32] = field(default_factory=lambda: np.zeros((3, 256), dtype=np.uint32))

    # Noise Prediction Modes
    channel_modes: npt.NDArray[np.uint8] = field(default_factory=lambda: np.zeros(3, dtype=np.uint8))

    # Extracted data (for verification)
    channels: Optional[Tuple] = None
    # Template selection modes (3 × n_shards)
    shard_modes: npt.NDArray[np.uint8] = field(default_factory=lambda: np.zeros((3, PROFILE_RGB.total_shards), dtype=np.uint8))

    # Final compressed payload for in-memory benchmarks
    payload: Optional[bytes] = field(default=None, repr=False)
    mode: str = "RGB"
    aad: float = 0.0

    @property
    def ratio(self) -> float:
        return self.comp_size / self.orig_size if self.orig_size > 0 else 1.0

    @property
    def pixel_count(self) -> int:
        return self.h * self.w

def extract_srb_metadata(shard_stats: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint16]:
    """ Determines the observed ZigZag symbol width per shard for PDF compaction. """
    n_shards = shard_stats.shape[1]
    widths = np.ones((3, n_shards), dtype=np.uint16)
    for c in range(3):
        for s in range(n_shards):
            hist = shard_stats[c, s]
            if np.any(hist):
                indices = np.where(hist > 0)[0]
                widths[c, s] = np.uint16(int(indices[-1]) + 1)
    return widths

@njit(parallel=True, fastmath=True, cache=True)
def normalize_shard_stats(shard_stats: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
    """
    Residual Normalization Transformation (BICC).
    Shifts centered histograms (base 128) to align the distribution peak to 0 in ZigZag space.
    """
    n_colors, n_shards, _ = shard_stats.shape
    aligned_stats = np.zeros((n_colors, n_shards, 256), dtype=np.uint32)
    m = 128 # Static BICC Bias
    
    for c in range(n_colors):
        for s in prange(n_shards):
            hist = shard_stats[c, s]
            if np.sum(hist) == 0: continue
            for centered_val in range(256):
                count = hist[centered_val]
                if count == 0: continue
                # Align peak to 0 before ZigZag mapping
                norm_res = centered_val - m
                z_aligned = int(to_zigzag(np.uint8(norm_res & 0xFF)))
                aligned_stats[c, s, z_aligned] += count
    return aligned_stats

@njit(fastmath=True, error_model='numpy', cache=True)
def calculate_channel_stats(hist: npt.NDArray[np.uint32]) -> Tuple[float, int, int]:
    """
    Derives O(256) metrics from a global channel histogram.
    Returns (Mean, Median, Mode).
    """
    total = np.sum(hist)
    if total == 0: return 0.0, 0, 0
    
    # 1. Mean
    s = 0.0
    for i in range(256):
        s += float(i) * hist[i]
    mean_val = s / float(total)
    
    # 2. Mode (Highest Frequency)
    mode_val = 0
    max_count = 0
    for i in range(256):
        if hist[i] > max_count:
            max_count = hist[i]
            mode_val = i
            
    # 3. Median (50th Percentile)
    acc = 0
    median_val = 0
    midpoint = (total + 1) // 2
    for i in range(256):
        acc += hist[i]
        if acc >= midpoint:
            median_val = i
            break
            
    return mean_val, median_val, mode_val


# --- Per-Shard Dispatch Tables ---

# Bitplane-width threshold: 90th-percentile ZigZag residual width over active
# shards.  p90 captures tail behaviour - bitplane needs the entire distribution
# to be narrow, not just the average.  Natural images have wide high-energy
# boundary shards that inflate the tail even when the median is low.
# Empirically: Tecnick p90 max = 95, DIV2K p90 min = 70.5 (at 1 Mpx+ gate).
# Threshold 85 gives 99% classification accuracy vs 96% for mean@53.
BITPLANE_H_THRESHOLD: float = 3.3          # Shannon Entropy Gating (bits/symbol)
BITPLANE_HIT_RATE_THRESHOLD: float = 0.20    # Minimum Zero-Residual Fraction
BITPLANE_P90_THRESHOLD: int = 175             # Max 90th-percentile ZigZag symbol width

ENABLE_DIAGNOSTICS: bool = False  # Production Gate
