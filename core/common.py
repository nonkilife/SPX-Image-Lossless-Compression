"""
ZPNG-CSDE v6.2 [Flexible-Shard Architecture]
Module: zpng_common
Role: Pillar 1 - Foundation & Protocol.
Description: Authoritative definitions for constants, sharding matrices, and header flags.
Architecture: Flexible Sharding Hub utilizing 3D Mapping LUTs for context ID derivation.

Logic Path:
```mermaid
graph TD
    In[Input: ag, bg, cg, Intensity] --> Grad[Calculate Gradients: dh, dv, v=max]
    Grad --> Hub[Flexible Sharding Hub]
    Hub --> Map[Lookup: ShardMap Tier, Intensity, Trend]
    Map --> CID[Context ID]
```
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange, uint8, uint16, uint32, uint64
from typing import Tuple, Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from .predictor import to_zigzag, from_zigzag, predict_med_standard

# --- 供 codec 與 rans 模組使用的經驗分布範本庫 (v6.6 Data-Driven) ---
# 這些模板是從 6185 個真實影像分片中透過 K-Means 聚類提取出的「靈魂分布」。
# 包含對稱分布與處理高反差邊緣的「非對稱分布」。
# --- 供 codec 與 rans 模組使用的經驗分布範本庫 (v6.6 Data-Driven) ---

def _build_empirical_templates() -> tuple[npt.NDArray[np.uint64], ...]:
    """ 
    V6.1 T3 Template Stack (30-Template Matrix).
    Derived from 6 PHOTOGRAPHIC CENTROIDS across 5 SIGMA SCALES [0.5, 0.75, 1.0, 1.25, 1.5].
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

EMPIRICAL_TEMPLATES: tuple[npt.NDArray[np.uint64], ...] = _build_empirical_templates()

# --- 1. Protocol Constants (Header Flags) ---
FLAG_RGBA: int        = 0x01
FLAG_SIMPLE: int      = 0x02  # Zstd-compressed Raw Pixels (No sharding)
FLAG_RAW: int         = 0x04  # Uncompressed Raw Pixels (No zstd)
FLAG_PASSTHROUGH: int = 0x08  # Original File Storage (PNG/JPG)
FLAG_GRAYSCALE: int   = 0x10  # Hardware-accelerated true monochrome bypassing CR
FLAG_COLOR_GSUB: int   = 0x20  # Adaptive Green-Subtract Transform (Smooth Image Optimization)
FLAG_BITPLANE: int    = 0x40  # 2D Bit-Context engine
FLAG_TEMPLATE_MODE: int = 0x80 # [v6.5] Use Laplace-Template entropy coding

# --- 2. Sharding Profile System [v5.7 Systematic Alignment] ---

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
    v_boundaries_rd_bd: npt.NDArray[np.uint8]
    intensity_segments: npt.NDArray[np.uint8]
    tiers_per_intensity: int
    hi_tier_boundary: int
    noise_shard_id: int  # -1 if no noise shard
    total_shards: int
    shard_map: npt.NDArray[np.uint8] # [v_level][intensity_idx][trend_idx]
    res_context_states: int = 1

# 2.1 Profile: BICC-Full (28-Shard Matrix)
V_BOUND_GR_FULL = np.array([0, 2, 4, 10, 255], dtype=np.uint8)
V_BOUND_RD_BD_FULL = np.array([0, 1, 3, 8, 255], dtype=np.uint8)
INTENSITY_SEG_FULL = np.array([0, 60, 190, 255], dtype=np.uint8)
TPI_FULL = (len(V_BOUND_GR_FULL) - 1) * 2 + 1
NSID_FULL = TPI_FULL * (len(INTENSITY_SEG_FULL) - 1)

def build_shard_map_he_rgb() -> npt.NDArray[np.uint8]:
    """ Builds isomorphic 3D map for HE-RGB (28 shards). 
        Layout: [5 Tiers][3 Intensity][2 Trends: 0:Straight, 1:Diag]
    """
    s_map = np.zeros((5, 3, 3), dtype=np.uint8) # Trend max 3 for safety
    tpi = 9
    for i in range(3):
        base = i * tpi
        # V=0 (Tier 0)
        s_map[0, i, :] = base + 0
        # V>0 (Tiers 1, 2, 3, 4)
        for v in range(1, 5):
            s_map[v, i, 0] = base + 1 + (v-1)*2 # Straight
            s_map[v, i, 1] = base + 2 + (v-1)*2 # Diag
    return s_map

# 2. Unified Profile Settings (v6.6)
V_BOUND_RGB = np.array([0, 1, 2, 4, 8, 16, 32, 255], dtype=np.uint8)
INTENSITY_SEG_RGB = np.array([0, 60, 190, 255], dtype=np.uint8)
TPI_RGB = 41 # Total 42 shards

def build_shard_map_universal_42() -> npt.NDArray[np.uint8]:
    """ Unified 42-shard Balanced Architecture (v6.6). """
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
    v_boundaries_rd_bd=V_BOUND_RGB,
    intensity_segments=INTENSITY_SEG_RGB,
    tiers_per_intensity=TPI_RGB + 1,
    hi_tier_boundary=5,
    noise_shard_id=-1,
    total_shards=42,
    shard_map=build_shard_map_universal_42()
)

# Legacy Compatibility Aliases (to minimize breaking changes in other files)
TOTAL_SHARDS = 42

ENABLE_DIAGNOSTICS: bool = False  # Production Gate

def get_shard_labels() -> List[str]:
    """ Automatically generates descriptive labels for all shards (Max 30). """
    labels = [f"Shard_{i}" for i in range(TOTAL_SHARDS)]
    # We could make this semantic, but for now generic index is safer for mixed modes
    return labels

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
    
    
    # Sharding
    shard_counts: npt.NDArray[np.uint32] = field(default_factory=lambda: np.zeros((3, TOTAL_SHARDS), dtype=np.uint32))
    shard_ptrs: Optional[Tuple] = None
    shard_stats: npt.NDArray[np.uint32] = field(default_factory=lambda: np.zeros((3, TOTAL_SHARDS, 256), dtype=np.uint32))
    shard_mins: npt.NDArray[np.uint8] = field(default_factory=lambda: np.zeros((3, TOTAL_SHARDS), dtype=np.uint8))
    shard_widths: npt.NDArray[np.uint16] = field(default_factory=lambda: np.zeros((3, TOTAL_SHARDS), dtype=np.uint16))
    
    # [v4.10.2] Channel Statistical Data (Global Histograms: Grn, RD, BD)
    channel_hists: npt.NDArray[np.uint32] = field(default_factory=lambda: np.zeros(3, dtype=np.uint32))
    
    # [v4.11.0] Noise Prediction Modes
    channel_modes: npt.NDArray[np.uint8] = field(default_factory=lambda: np.zeros(3, dtype=np.uint8))
    
    # Extracted data (for verification)
    channels: Optional[Tuple] = None
    # [v6.5] Median Normalization Metrics (3x42 uint8)
    shard_medians: npt.NDArray[np.uint8] = field(default_factory=lambda: np.zeros((3, TOTAL_SHARDS), dtype=np.uint8))
    # [v6.6] Template Selection Modes (3x42 uint8)
    shard_modes: npt.NDArray[np.uint8] = field(default_factory=lambda: np.zeros((3, TOTAL_SHARDS), dtype=np.uint8))
    
    # [v2.15] Memory Optimization: Store final payload for in-memory benchmarks
    payload: Optional[bytes] = field(default=None, repr=False)
    mode: str = "RGB"
    aad: float = 0.0

    @property
    def ratio(self) -> float:
        return self.comp_size / self.orig_size if self.orig_size > 0 else 1.0

    @property
    def pixel_count(self) -> int:
        return self.h * self.w

def extract_srb_metadata(shard_stats: npt.NDArray[np.uint32]) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint16]]:
    """ 
    [v6.1 BICC] Metadata Analytical Kernel.
    Determines the observed symbol width for PDF compaction. 
    Maintains zero-centered shift protocol (mins = 0) to preserve T3 template compatibility.
    """
    n_shards = shard_stats.shape[1]
    widths = np.ones((3, n_shards), dtype=np.uint16) 
    mins = np.zeros((3, n_shards), dtype=np.uint8) 
    for c in range(3):
        for s in range(n_shards):
            hist = shard_stats[c, s]
            if np.sum(hist) > 0:
                indices = np.where(hist > 0)[0]
                if len(indices) > 0:
                    widths[c, s] = np.uint16(int(np.max(indices)) + 1)
    return mins, widths

@njit(parallel=True, fastmath=True, cache=True)
def apply_median_to_stats(shard_stats: npt.NDArray[np.uint32], medians: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint32]:
    """
    [v6.1 BICC] Median Normalization Transformation.
    Shifts centered histograms to align the distribution peak (median) to 0 in ZigZag space.
    This ensures that different shards with similar variances but different bias offsets 
    can be modeled by the same static Laplacian template.
    """
    n_colors, n_shards, _ = shard_stats.shape
    aligned_stats = np.zeros((n_colors, n_shards, 256), dtype=np.uint32)
    
    for c in prange(n_colors):
        for s in range(n_shards):
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
    [v4.10.2] Derived O(256) Metrics from Global Channel Histograms.
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
    midpoint = total // 2
    for i in range(256):
        acc += hist[i]
        if acc > midpoint:
            median_val = i
            break
            
    return mean_val, median_val, mode_val



# (predict_med_extreme removed for pipeline flattening)

# Global Immutable Dispatchers [v6.2 Systematic Refactor]
# Note: V_OFF_LUTs are deprecated; dispatching is now boundary-driven for flexibility.

@njit(fastmath=True, error_model='numpy', inline='always', cache=True)
def get_trend_idx(ag: np.uint8, bg: np.uint8, cg: np.uint8) -> np.uint8:
    """ Unified Trend Extraction (v6.6): Rising/Falling/Mix logic. """
    rising = np.uint8(ag > cg) * np.uint8(bg > cg)
    falling = np.uint8(ag < cg) * np.uint8(bg < cg)
    return np.uint8(falling + 2 * (1 - (rising + falling)))

@njit(fastmath=True, error_model='numpy', inline='always', cache=True)
def get_context_id_flexible(ag: np.uint8, bg: np.uint8, cg: np.uint8, intensity: np.uint8, 
                          shard_map: npt.NDArray[np.uint8], boundaries: npt.NDArray[np.uint8],
                          i_segs: npt.NDArray[np.uint8], nsid: int) -> np.uint8:
    """ ZPNG v6.2 Flexible Dispatcher: Strength x Intensity x Trend [Branchless Optimized]. """
    dh: np.int32 = abs(np.int32(ag) - np.int32(cg))
    dv: np.int32 = abs(np.int32(bg) - np.int32(cg))
    v: np.uint32 = np.uint32(max(dh, dv))
    
    # [1] Intensity Logic (Branchless Segment Discovery)
    i_idx = np.uint8(intensity > i_segs[1]) + np.uint8(intensity > i_segs[2])
    
    # [2] Strength Logic (Branchless V-Tier Comparison Summation)
    v_tier = np.uint8(v > 0)
    for i in range(1, len(boundaries) - 1):
        v_tier += np.uint8(v > boundaries[i])
            
    # [3] Trend Logic
    t_idx = get_trend_idx(ag, bg, cg)
    
    # [4] Map Lookup
    base_cid = shard_map[v_tier, i_idx, t_idx]
    
    # [5] Noise Shard Overrule (Branchless)
    ns_hit = np.uint8(nsid >= 0) * np.uint8(dh > 12) * np.uint8(dv > 12)
    
    return np.uint8((1 - ns_hit) * base_cid + ns_hit * max(0, nsid))

# --- End of Flexible Sharding Hub ---
