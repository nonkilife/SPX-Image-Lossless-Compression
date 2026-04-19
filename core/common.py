"""
ZPNG-CSDE [Flexible Shard Architecture]
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
    %% Packing Format: [V_Tier:5 bits] | [Trend:2 bits] | [Noise:1 bit]
```
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange, uint8
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
from .predictor import to_zigzag, from_zigzag, predict_med_standard

# Empirical distribution template library for codec and rans modules.
# Extracted from 6185 real image shards via K-Means clustering, including symmetric and asymmetric distributions.

def _build_empirical_templates() -> Tuple[npt.NDArray[np.uint64], ...]:
    """
    30-template matrix: 6 photographic centroids × 5 sigma scales [0.5, 0.75, 1.0, 1.25, 1.5].
    Derived from 6185 real image shards via K-Means clustering.
    """
    P_LIST = [
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,5,5,6,6,7,8,9,10,11,12,14,16,18,22,27,32,39,50,66,86,115,170,233,305,401,656,399,305,234,171,116,87,67,51,39,32,27,22,19,16,14,12,11,10,9,8,7,6,6,5,5,4,4,3,3,3,3,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,4,4,5,6,8,11,15,21,31,48,82,167,597,2016,606,179,90,52,33,21,14,11,8,7,5,5,4,3,3,2,2,2,2,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,6,6,7,7,8,9,10,11,12,13,14,16,18,20,22,25,29,32,36,43,50,56,64,77,90,104,123,152,182,218,264,501,262,218,183,152,122,104,90,77,64,55,49,42,36,32,29,24,21,20,18,15,14,13,11,10,9,9,8,7,7,7,6,5,5,5,4,4,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,4,5,6,7,9,11,12,15,19,23,28,33,40,46,55,63,74,85,96,109,120,132,144,155,167,181,203,329,196,178,167,157,145,133,122,109,98,86,74,65,55,47,40,33,28,23,19,16,13,11,9,7,6,5,4,4,3,3,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,6,6,6,6,7,7,7,8,8,8,9,9,10,10,11,11,12,13,14,14,15,16,17,18,19,21,22,24,25,27,29,31,33,36,39,42,45,48,53,57,62,68,73,81,87,96,106,119,134,151,311,150,134,118,107,97,88,81,75,68,62,57,53,49,45,42,39,36,33,31,29,27,25,23,22,20,19,18,17,16,15,14,13,12,12,11,11,10,10,9,9,8,8,8,7,7,7,6,6,6,5,5,5,5,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64),
        np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3,4,4,5,5,6,6,7,8,9,10,11,13,15,18,22,27,35,47,66,102,168,296,581,1114,572,296,170,103,67,48,36,28,22,18,15,13,11,10,9,8,7,6,6,5,4,4,4,3,3,3,2,2,2,2,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint64)
    ]
    SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
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
FLAG_BITPLANE: int    = 0x40  # 2D Bit-Context engine
FLAG_TEMPLATE_MODE: int = 0x80 # Use Laplace-Template entropy coding

# --- 2. Sharding Profile System ---

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

# --- 2a. Default Profile Settings ---
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

# Used internally by ZpngResult defaults and get_shard_labels; not imported by other modules.
TOTAL_SHARDS = 42

# --- Per-Shard Dispatch Tables ---

# Bitplane-width threshold: 90th-percentile ZigZag residual width over active
# shards.  p90 captures tail behaviour - bitplane needs the entire distribution
# to be narrow, not just the average.  Natural images have wide high-energy
# boundary shards that inflate the tail even when the median is low.
# Empirically: Tecnick p90 max = 95, DIV2K p90 min = 70.5 (at 1 Mpx+ gate).
# Threshold 85 gives 99% classification accuracy vs 96% for mean@53.
BITPLANE_WIDTH_THRESHOLD: int = 85

# Minimum pixel count (h*w) for bitplane rANS to be considered.
# Bitplane carries a fixed per-channel table overhead (~40-60 KB compressed)
# that dominates on small images.  Kodak (0.39 Mpx) loses +10-17%;
# images below ~1 Mpx consistently suffer regardless of mean_width.
BITPLANE_MIN_PIXELS: int = 1_000_000

ENABLE_DIAGNOSTICS: bool = False  # Production Gate

def get_shard_labels(n_shards: Optional[int] = None) -> List[str]:
    """ Generates generic index labels for all shards in the given (or current) profile. """
    if n_shards is None:
        n_shards = PROFILE_RGB.total_shards
    return [f"Shard_{i}" for i in range(n_shards)]

SHARD_LABELS: List[str] = get_shard_labels()


# --- 3. Data Structures ---

@dataclass
class ZpngResult:
    """ Unified container for ZPNG compression/decompression metrics. """
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
    # Median normalization metrics (3 × n_shards)
    shard_medians: npt.NDArray[np.uint8] = field(default_factory=lambda: np.zeros((3, PROFILE_RGB.total_shards), dtype=np.uint8))
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
def apply_median_to_stats(shard_stats: npt.NDArray[np.uint32], medians: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint32]:
    """
    Median Normalization Transformation (BICC).
    Shifts centered histograms to align the distribution peak (median) to 0 in ZigZag space.
    This ensures that different shards with similar variances but different bias offsets
    can be modeled by the same static Laplacian template.
    """
    n_colors, n_shards, _ = shard_stats.shape
    aligned_stats = np.zeros((n_colors, n_shards, 256), dtype=np.uint32)
    
    for c in range(n_colors):
        for s in prange(n_shards):
            m = int(medians[c, s])
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
    Derived O(256) metrics from a global channel histogram.
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



# --- High-Performance Context Dispatcher LUTs ---
SPATIAL_TRANS_LUT = np.zeros((511, 511), dtype=np.uint8)
INTENSITY_LUT = np.zeros(256, dtype=np.uint8)
FINAL_DISPATCH_LUT = np.zeros(1024, dtype=np.uint8)  # index = (packed<<2)|i_idx

def initialize_luts_python(v_bounds, i_segs, shard_map, nsid: int):
    """
    Fills global LUTs with precomputed context features.

    Engineering Logic:
    1. Spatial Consistency: Uses (ag-cg) and (bg-cg) to derive a 2D gradient vector.
       By subtracting the top-left neighbor (cg), we isolate the local 'slope'
       from the absolute pixel offset, ensuring the same context is recognized
       regardless of the global brightness level.
    2. Zero-Latency Dispatch: Pre-packs V (Variance), T (Trend), and N (Noise)
       into a single uint8. This allows get_context_id_fast to perform
       feature extraction using bit-shifts instead of branching logic.
    """
    # 1. Intensity LUT — generalized: one threshold crossing per i_segs[1:-1] boundary
    i_arr = np.arange(256, dtype=np.uint8)
    i_result = np.zeros(256, dtype=np.uint8)
    for thr in i_segs[1:-1]:
        i_result += (i_arr > int(thr)).astype(np.uint8)
    INTENSITY_LUT[:] = i_result

    # 2. Spatial Transition LUT (511x511, vectorized)
    d = np.arange(-255, 256, dtype=np.int16)
    DA, DB = np.meshgrid(d, d, indexing='ij')   # (511, 511)

    # Strength (V): count how many boundaries are exceeded
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

    # Packing: [V:5][T:2][N:1]
    SPATIAL_TRANS_LUT[:] = ((v_tier << 3) | (t_idx << 1) | ns_hit).astype(np.uint8)

    # 3. Final Dispatch LUT — collapses branch + 3D shard_map lookup into a single
    #    1024-byte table. index = (packed<<2)|i_idx; result = shard_id.
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

# Auto-initialize with default Universal-42 bounds (Internal Cache)
_LAST_V_BOUNDS = np.zeros(8, dtype=np.uint8)
_LAST_I_SEGS = np.zeros(4, dtype=np.uint8)
_LAST_NSID: int = -2  # sentinel: forces first-run initialization

def sync_luts_if_needed(v_bounds, i_segs, shard_map, nsid: int):
    """
    Ensures global LUTs match the requested profile.
    Must be called from Python context before entering JIT kernels.
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


@njit(fastmath=True, cache=True)
def get_context_id_fast(ag: uint8, bg: uint8, cg: uint8, intensity: uint8,
                        shard_map: npt.NDArray[np.uint8], nsid: int) -> uint8:
    """
    Triple LUT dispatch: SPATIAL_TRANS_LUT → INTENSITY_LUT → FINAL_DISPATCH_LUT.
    shard_map/nsid are baked into FINAL_DISPATCH_LUT at sync time; kept in
    signature for caller compatibility.
    """
    packed = SPATIAL_TRANS_LUT[int(ag) - int(cg) + 255, int(bg) - int(cg) + 255]
    i_idx  = INTENSITY_LUT[intensity]
    return FINAL_DISPATCH_LUT[(packed << 2) | i_idx]

# --- End of Flexible Sharding Hub ---
