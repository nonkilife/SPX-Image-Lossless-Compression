"""
SPX v8.3.2-stable [Stateless Sharding Hub]
Module: sharding
Role: Pillar 4 - Strategic Partitioning (BICC Integration).

Architecture:
1. Fused RCT/Pass 1: Combined color decorrelation and statistical profiling in a single raster pass.
2. Stateless Dispatch: O(1) context derivation using profile-owned LUTs, eliminating global state.
3. Zero-Copy Design: Minimizes intermediate DRAM traffic by utilizing JIT-fused kernels.

Logic Flow:
```mermaid
graph TD
    Input[RGB/RGBA Raw] --> Fused[fused_rct_p1: Decorrelate + Profile + Cache]
    Fused -->|Pass 1 Result| Dataclass[Pass1Result: Structured Metadata]
    Dataclass -->|Standard Mode| Pass2[shard_pass_2: Gather Residuals]
    Pass2 -->|Final Payload| Buffer[ShardBuffer: Sharded Bitstream Ready]

    Dataclass -->|Bitplane Mode| BPlane[rans_bitplane: Contextual Entropy Coding]
```
"""

import numpy as np
import numpy.typing as npt
import spx_rans as _rs
from dataclasses import dataclass, field
from typing import Tuple, Optional

__version__ = "8.3.2"

__all__ = [
    'SpxResult',
    'ShardProfile',
    'Pass1Result',
    'ShardBuffer',
    'PROFILE_RGB',
    'extract_srb_metadata',
    'execute_sharding_stateless',
    'reconstruct_shards_rgb',
]

# --- 1. Data Structures ---

@dataclass
class SpxResult:
    """ Unified container for SPX compression/decompression metrics. """
    enc_time: float = 0.0
    dec_time: float = 0.0
    h: int = 0
    w: int = 0
    is_rgba: bool = False
    comp_size: int = 0
    orig_size: int = 0
    hits: npt.NDArray[np.uint32] = field(default_factory=lambda: np.zeros(4, dtype=np.uint32))
    res_sums: npt.NDArray[np.uint64] = field(default_factory=lambda: np.zeros(4, dtype=np.uint64))
    shard_counts: npt.NDArray[np.uint32] = field(default_factory=lambda: np.empty((3, 0), dtype=np.uint32))
    shard_ptrs: Optional[Tuple[int, ...]] = None
    shard_stats: npt.NDArray[np.uint32] = field(default_factory=lambda: np.empty((3, 0, 256), dtype=np.uint32))
    shard_widths: npt.NDArray[np.uint16] = field(default_factory=lambda: np.empty((3, 0), dtype=np.uint16))
    channel_hists: npt.NDArray[np.uint32] = field(default_factory=lambda: np.zeros((3, 256), dtype=np.uint32))
    channel_modes: npt.NDArray[np.uint8] = field(default_factory=lambda: np.zeros(3, dtype=np.uint8))
    channels: Optional[Tuple[npt.NDArray[np.uint8], ...]] = None
    shard_modes: npt.NDArray[np.uint8] = field(default_factory=lambda: np.empty((3, 0), dtype=np.uint8))
    payload: Optional[bytes] = field(default=None, repr=False)
    mode: str = "RGB"

    @property
    def ratio(self) -> float:
        return self.comp_size / self.orig_size if self.orig_size > 0 else 1.0

    @property
    def pixel_count(self) -> int:
        return self.h * self.w

def extract_srb_metadata(shard_stats: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint16]:
    """ 
    Determines the observed ZigZag symbol width per shard for PDF compaction. 
    This identifies the highest used symbol in each shard to prune the tail of 
    frequency tables during serialization.
    """
    n_channels, n_shards = shard_stats.shape[0], shard_stats.shape[1]
    widths = np.ones((n_channels, n_shards), dtype=np.uint16)
    for c in range(n_channels):
        for s in range(n_shards):
            hist = shard_stats[c, s]
            indices = np.where(hist > 0)[0]
            if indices.size > 0:
                widths[c, s] = np.uint16(int(indices[-1]) + 1)
    return widths

@dataclass(frozen=True)
class ShardProfile:
    """
    Authoritative physical architecture defining how context boundaries segment statistical space.
    Encapsulates its own precomputed LUTs for stateless dispatch.
    """
    name: str
    v_boundaries_gr: npt.NDArray[np.uint8]
    intensity_segments: npt.NDArray[np.uint8]
    noise_shard_id: int
    total_shards: int
    shard_map: npt.NDArray[np.uint8]
    spatial_lut: npt.NDArray[np.uint8]
    intensity_lut: npt.NDArray[np.uint8]
    dispatch_lut: npt.NDArray[np.uint8]

@dataclass(frozen=True)
class Pass1Result:
    """
    Primary result of the Fused Pass 1 kernel.
    Encapsulates both the decorrelated channels and the statistical profiles
    required for either Sharded rANS or Bitplane coding.
    """
    gr_ch_p: npt.NDArray[np.uint8]        # Padded Green/Luma channel (h+2, w+2)
    rd_ch_p: npt.NDArray[np.uint8]        # Padded Red-diff channel (h+2, w+2)
    bd_ch_p: npt.NDArray[np.uint8]        # Padded Blue-diff channel (h+2, w+2)
    a_ch: npt.NDArray[np.uint8]           # Raw Alpha channel (h, w)
    shard_counts: npt.NDArray[np.uint32]  # Counts per shard [3, n_shards]
    shard_stats: npt.NDArray[np.uint32]   # Histograms per shard [3, n_shards, 256]
    shard_offsets: npt.NDArray[np.uint32] # Bitstream base offsets per shard [3, n_shards]
    row_global_offsets: npt.NDArray[np.uint32] # Linear pointers into payloads [h, 3, n_shards]
    hits: npt.NDArray[np.uint32]          # Total hit-rate metric (zero-residuals)
    sums: npt.NDArray[np.uint64]          # Absolute residual sums (entropy heuristic)
    channel_hists: npt.NDArray[np.uint32] # Global channel histograms (un-sharded)
    res_cached: Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]] # Cached residuals for Bitplane
    a_metrics: Tuple[np.uint64, np.float64] # Alpha-specific hit/sum metrics

@dataclass
class ShardBuffer:
    """
    Unified container for partitioned shard residuals and metadata.
    """
    gr_payload: npt.NDArray[np.uint8]
    rd_payload: npt.NDArray[np.uint8]
    bd_payload: npt.NDArray[np.uint8]
    a_payload: npt.NDArray[np.uint8]
    counts: npt.NDArray[np.uint32]
    stats: npt.NDArray[np.uint32]
    offsets: npt.NDArray[np.uint32]
    row_offsets: npt.NDArray[np.uint32]
    hits: npt.NDArray[np.uint32]
    sums: npt.NDArray[np.uint64]
    a_metrics: Tuple[np.uint64, np.float64] = (np.uint64(0), 0.0)

# --- 2. Profile Generation Logic ---

def precompute_luts(v_bounds: npt.NDArray[np.uint8],
                    i_segs: npt.NDArray[np.uint8],
                    shard_map: npt.NDArray[np.uint8],
                    nsid: int) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """ 
    Generates profile-specific LUTs for context features.
    These LUTs allow the Rust kernels to derive context IDs in O(1) time.
    
    1. intensity_lut: Maps pixel intensity [0-255] to a coarse segment (Dark, Mid, Light).
    2. spatial_lut: Maps local gradient pairs (da, db) to (V-Tier, Trend, Noise) features.
    3. dispatch_lut: Maps the packed features to a final Shard ID.
    """
    # [Intensity LUT]
    i_lut = np.zeros(256, dtype=np.uint8)
    i_arr = np.arange(256, dtype=np.uint8)
    for thr in i_segs[1:-1]:
        i_lut += (i_arr > int(thr)).astype(np.uint8)

    # [Spatial LUT] - 511x511 grid covering all possible (A-C, B-C) differences.
    s_lut = np.zeros((511, 511), dtype=np.uint8)
    d = np.arange(-255, 256, dtype=np.int16)
    DA, DB = np.meshgrid(d, d, indexing='ij')

    # V-Tier: Represents the maximum absolute gradient magnitude.
    v_mag = np.maximum(np.abs(DA), np.abs(DB))
    v_tier = (v_mag > 0).astype(np.uint8)
    for i in range(1, len(v_bounds) - 1):
        v_tier += (v_mag > int(v_bounds[i])).astype(np.uint8)

    # Trend: Detects directionality (Rising, Falling, or Flat/Complex).
    rising  = ((DA > 0) & (DB > 0)).astype(np.uint8)
    falling = ((DA < 0) & (DB < 0)).astype(np.uint8)
    t_idx   = (falling + 2 * (1 - rising - falling)).astype(np.uint8)

    # Noise: Identifies high-energy outliers where both neighbors deviate significantly.
    ns_hit = ((np.abs(DA) > 12) & (np.abs(DB) > 12)).astype(np.uint8)
    
    # Pack into a single byte for the Dispatch LUT.
    s_lut[:] = ((v_tier << 3) | (t_idx << 1) | ns_hit).astype(np.uint8)

    # [Dispatch LUT] - Final mapping to Shard IDs.
    d_lut = np.zeros((256, 4), dtype=np.uint8)
    n_v, n_t = shard_map.shape[0], shard_map.shape[2]

    for pk in range(256):
        vt, ti, ns = pk >> 3, (pk >> 1) & 0x03, pk & 0x01
        for ii in range(3): # For each intensity segment
            if ns != 0 and nsid >= 0: d_lut[pk, ii] = np.uint8(nsid)
            elif vt < n_v and ti < n_t: d_lut[pk, ii] = shard_map[vt, ii, ti]
            else: d_lut[pk, ii] = np.uint8(nsid if nsid >= 0 else 0)

    return s_lut, i_lut, d_lut


# --- 3. Universal-42 Profile ---
# The "Universal-42" is the standard profile for photographic content.
# It segments the image into 42 statistical buckets (shards) based on 
# edge strength (8 tiers), intensity (3 segments), and trend (3 types).

V_BOUND_RGB = np.array([0, 1, 2, 4, 8, 16, 32, 255], dtype=np.uint8)
INTENSITY_SEG_RGB = np.array([0, 60, 190, 255], dtype=np.uint8)

def build_shard_map_universal_42() -> npt.NDArray[np.uint8]:
    """
    Defines the logical mapping from (V-Tier, Intensity, Trend) to Shard ID.
    - Shards 0-2: Ultra-flat (Intensity only)
    - Shards 3-29: Low-Mid complexity (V-Tiers 1-3, full Intensity x Trend grid)
    - Shards 30-41: High complexity (V-Tiers 4-7, Trend only to reduce sparsity)
    """
    s_map = np.zeros((8, 3, 3), dtype=np.uint8)
    for i in range(3): s_map[0, i, :] = i
    for v in range(1, 4):
        for i in range(3):
            base = 3 + (v-1) * 9 + i * 3
            s_map[v, i, 0], s_map[v, i, 1], s_map[v, i, 2] = base, base + 1, base + 2
    for v in range(4, 8):
        for i in range(3):
            base = 30 + (v-4) * 3
            s_map[v, i, 0], s_map[v, i, 1], s_map[v, i, 2] = base, base + 1, base + 2
    return s_map

_shard_map_rgb = build_shard_map_universal_42()
_s_lut_rgb, _i_lut_rgb, _d_lut_rgb = precompute_luts(V_BOUND_RGB, INTENSITY_SEG_RGB, _shard_map_rgb, -1)

PROFILE_RGB = ShardProfile(
    name="Universal-42",
    v_boundaries_gr=V_BOUND_RGB,
    intensity_segments=INTENSITY_SEG_RGB,
    noise_shard_id=-1,
    total_shards=42,
    shard_map=_shard_map_rgb,
    spatial_lut=_s_lut_rgb,
    intensity_lut=_i_lut_rgb,
    dispatch_lut=_d_lut_rgb
)

# --- 4. Rust-backed Sharding Kernels ---
# The following functions are Python shims that delegate to high-performance 
# Rust kernels. Note the use of np.ascontiguousarray() to ensure memory safety 
# when crossing the FFI boundary.

def shard_pass_2_rgb_stateless(h: int, w: int,
                                gr_ch: npt.NDArray[np.uint8], rd_ch: npt.NDArray[np.uint8], bd_ch: npt.NDArray[np.uint8],
                                row_global_offsets: npt.NDArray[np.uint32],
                                shard_gr: npt.NDArray[np.uint8], shard_rd: npt.NDArray[np.uint8], shard_bd: npt.NDArray[np.uint8],
                                s_lut: npt.NDArray[np.uint8], i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]) -> None:
    """ Rust implementation of the RGB residual gathering pass. """
    _rs.p2_rgb(h, w,
               np.ascontiguousarray(gr_ch, dtype=np.uint8),
               np.ascontiguousarray(rd_ch, dtype=np.uint8),
               np.ascontiguousarray(bd_ch, dtype=np.uint8),
               np.ascontiguousarray(row_global_offsets, dtype=np.uint32),
               np.ascontiguousarray(shard_gr, dtype=np.uint8),
               np.ascontiguousarray(shard_rd, dtype=np.uint8),
               np.ascontiguousarray(shard_bd, dtype=np.uint8),
               np.ascontiguousarray(s_lut, dtype=np.uint8),
               np.ascontiguousarray(i_lut, dtype=np.uint8),
               np.ascontiguousarray(d_lut, dtype=np.uint8))

def shard_pass_2_gray_stateless(h: int, w: int,
                                 gr_ch: npt.NDArray[np.uint8],
                                 row_global_offsets: npt.NDArray[np.uint32],
                                 shard_gr: npt.NDArray[np.uint8],
                                 s_lut: npt.NDArray[np.uint8], i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]) -> None:
    """ Rust implementation of the Grayscale residual gathering pass. """
    _rs.p2_gray(h, w,
                np.ascontiguousarray(gr_ch, dtype=np.uint8),
                np.ascontiguousarray(row_global_offsets, dtype=np.uint32),
                np.ascontiguousarray(shard_gr, dtype=np.uint8),
                np.ascontiguousarray(s_lut, dtype=np.uint8),
                np.ascontiguousarray(i_lut, dtype=np.uint8),
                np.ascontiguousarray(d_lut, dtype=np.uint8))

def reconstruct_shards_rgb(h: int, w: int, res_gr: npt.NDArray[np.uint8], res_rd: npt.NDArray[np.uint8],
                           res_bd: npt.NDArray[np.uint8], off_gr: npt.NDArray[np.uint32],
                           off_rd: npt.NDArray[np.uint32], off_bd: npt.NDArray[np.uint32],
                           s_lut: npt.NDArray[np.uint8],
                           i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]):
    """ Rust implementation of the inverse sharding pass (scatter residuals back to image). """
    return _rs.reconstruct_shards_rgb(
        h, w,
        np.ascontiguousarray(res_gr, dtype=np.uint8),
        np.ascontiguousarray(res_rd, dtype=np.uint8),
        np.ascontiguousarray(res_bd, dtype=np.uint8),
        np.ascontiguousarray(off_gr, dtype=np.uint32),
        np.ascontiguousarray(off_rd, dtype=np.uint32),
        np.ascontiguousarray(off_bd, dtype=np.uint32),
        np.ascontiguousarray(s_lut, dtype=np.uint8),
        np.ascontiguousarray(i_lut, dtype=np.uint8),
        np.ascontiguousarray(d_lut, dtype=np.uint8))

def execute_sharding_stateless(h: int, w: int, p1: Pass1Result, profile: ShardProfile, is_grayscale: bool) -> ShardBuffer:
    """
    Fills ShardBuffer using the stateless Pass 2 gather kernels.
    Unlike Pass 1, this phase is 'Gather-Only' — it uses the pointers and offsets
    pre-calculated during Pass 1 to write residuals directly into their respective
    shard bitstreams without re-scanning or complex branching.
    """
    s_lut, i_lut, d_lut = profile.spatial_lut, profile.intensity_lut, profile.dispatch_lut
    s_gr = np.zeros(int(p1.shard_counts[0].sum()), dtype=np.uint8)
    if is_grayscale:
        shard_pass_2_gray_stateless(h, w, p1.gr_ch_p, p1.row_global_offsets, s_gr, s_lut, i_lut, d_lut)
        s_rd, s_bd = np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)
    else:
        s_rd, s_bd = np.zeros(int(p1.shard_counts[1].sum()), dtype=np.uint8), np.zeros(int(p1.shard_counts[2].sum()), dtype=np.uint8)
        shard_pass_2_rgb_stateless(h, w, p1.gr_ch_p, p1.rd_ch_p, p1.bd_ch_p, p1.row_global_offsets, s_gr, s_rd, s_bd, s_lut, i_lut, d_lut)
    return ShardBuffer(s_gr, s_rd, s_bd, p1.res_cached[3], p1.shard_counts, p1.shard_stats, p1.shard_offsets, p1.row_global_offsets, p1.hits, p1.sums, p1.a_metrics)
