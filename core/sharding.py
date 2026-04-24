"""
SPX [Stateless Sharding Hub]
Module: sharding
Role: Pillar 4 - Strategic Partitioning.
Description: Definitions for shard profiles, O(1) context derivation, and shard data containers.
Architecture: Profile-owned LUTs for zero-recompilation context derivation and unified ShardBuffer.

Design Philosophy: Configuration-as-Data
----------------------------------------
This module isolates the mathematical design of shards (boundaries, mapping matrices) 
from the execution logic. By centralizing all profiles here, we enable rapid 
switching between different segmentation strategies without modifying the core 
context derivation kernels.

Logic Flow:
```mermaid
graph TD
    Input[Input Channels: Gr, Rd, Bd, A] --> Mode{is_grayscale?}
    Mode -- Yes --> S1G[Pass 1 Gray: Profiling]
    Mode -- No --> S1R[Pass 1 RGB: Profiling]
    
    S1G & S1R --> Meta[Stats: counts, hists, row_offsets]
    Meta --> Alloc[Buffer Allocation: s_gr, s_rd, s_bd]
    
    Alloc --> S2G[Pass 2 Gray: Payload Creation]
    Alloc --> S2R[Pass 2 RGB: Payload Creation]
    
    S2G & S2R --> Output[ShardBuffer Container]
```
"""

import numpy as np
import numpy.typing as npt
from numba import njit, uint8, prange, uint32, uint16, uint64
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .predictor import (ZIGZAG_LUT, IZIGZAG_LUT, selected_predictor, BICC_ZIGZAG_LUT, from_zigzag)

# --- 1. Data Structures ---

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
    
    # Sharding Metadata (Agnostic to active profile until populated)
    shard_counts: npt.NDArray[np.uint32] = field(default_factory=lambda: np.empty((3, 0), dtype=np.uint32))
    shard_ptrs: Optional[Tuple[int, ...]] = None
    shard_stats: npt.NDArray[np.uint32] = field(default_factory=lambda: np.empty((3, 0, 256), dtype=np.uint32))
    shard_widths: npt.NDArray[np.uint16] = field(default_factory=lambda: np.empty((3, 0), dtype=np.uint16))

    # Channel Statistical Data (Global Histograms: Grn, RD, BD)
    channel_hists: npt.NDArray[np.uint32] = field(default_factory=lambda: np.zeros((3, 256), dtype=np.uint32))

    # Noise Prediction Modes
    channel_modes: npt.NDArray[np.uint8] = field(default_factory=lambda: np.zeros(3, dtype=np.uint8))

    # Extracted data (for verification)
    channels: Optional[Tuple[npt.NDArray[np.uint8], ...]] = None
    # Template selection modes (3 × n_shards)
    shard_modes: npt.NDArray[np.uint8] = field(default_factory=lambda: np.empty((3, 0), dtype=np.uint8))

    # Final compressed payload for in-memory benchmarks
    payload: Optional[bytes] = field(default=None, repr=False)
    mode: str = "RGB"

    @property
    def ratio(self) -> float:
        return self.comp_size / self.orig_size if self.orig_size > 0 else 1.0

    @property
    def pixel_count(self) -> int:
        return self.h * self.w

def extract_srb_metadata(shard_stats: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint16]:
    """ Determines the observed ZigZag symbol width per shard for PDF compaction. """
    n_channels, n_shards = shard_stats.shape[0], shard_stats.shape[1]
    widths = np.ones((n_channels, n_shards), dtype=np.uint16)
    for c in range(n_channels):
        for s in range(n_shards):
            hist = shard_stats[c, s]
            # [v8.1.0] Fix potential IndexError by checking indices size before access
            indices = np.where(hist > 0)[0]
            if indices.size > 0:
                widths[c, s] = np.uint16(int(indices[-1]) + 1)
    return widths

@njit(cache=True)
def normalize_shard_stats(shard_stats: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
    """ 
    Residual Normalization Transformation (BICC).
    -------------------------------------------
    Converts raw prediction residuals (which may be signed/biased) into 
    positive entropy symbols using BICC_ZIGZAG_LUT.
    
    Logic:
    1. Input histograms are indexed by (val - 128) & 0xFF.
    2. BICC_ZIGZAG_LUT maps these indices directly to ZigZag symbols.
    3. Output histograms are indexed by normalized ZigZag symbols.
    
    [v8.3.1] Passthrough: Histograms are now pre-normalized to ZigZag space during Pass 1 
    profiling to eliminate this separate O(N) scan.
    """
    return shard_stats

@njit(fastmath=True, error_model='numpy', cache=True)
def calculate_channel_stats(hist: npt.NDArray[np.uint32]) -> int:
    """ Returns the Mode (most frequent value) of the channel distribution. """
    if np.sum(hist) == 0: return 0
    return int(np.argmax(hist))

@dataclass(frozen=True)
class ShardProfile:
    """ 
    Authoritative physical architecture defining how context boundaries segment statistical space.
    Encapsulates its own precomputed LUTs for stateless dispatch.
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

@dataclass
class ShardBuffer:
    """ 
    Unified container for partitioned shard residuals and metadata. 
    Reduces parameter bloat in sharding and entropy coding kernels.
    """
    gr_payload: npt.NDArray[np.uint8]
    rd_payload: npt.NDArray[np.uint8]
    bd_payload: npt.NDArray[np.uint8]
    a_payload: npt.NDArray[np.uint8]
    counts: npt.NDArray[np.uint32]     # [3, n_shards]
    stats: npt.NDArray[np.uint32]      # [3, n_shards, 256]
    offsets: npt.NDArray[np.uint32]    # [3, n_shards]
    row_offsets: npt.NDArray[np.uint32] # [h, 3, n_shards]
    hits: npt.NDArray[np.uint32]       # [3]
    sums: npt.NDArray[np.uint64]       # [3]
    a_metrics: Tuple[np.uint64, np.float64] = (np.uint64(0), 0.0)

# --- 2. Profile Generation Logic ---

def precompute_luts(v_bounds: npt.NDArray[np.uint8], 
                    i_segs: npt.NDArray[np.uint8], 
                    shard_map: npt.NDArray[np.uint8], 
                    nsid: int) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """ Generates profile-specific LUTs for context features. """
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
    n_v, n_t = shard_map.shape[0], shard_map.shape[2]
    
    for pk in range(256):
        vt, ti, ns = pk >> 3, (pk >> 1) & 0x03, pk & 0x01
        for ii in range(3):
            if ns != 0 and nsid >= 0: d_lut[pk, ii] = np.uint8(nsid)
            elif vt < n_v and ti < n_t: d_lut[pk, ii] = shard_map[vt, ii, ti]
            else: d_lut[pk, ii] = np.uint8(nsid if nsid >= 0 else 0)
    
    return s_lut, i_lut, d_lut


# --- 3. Universal-42 Profile ---

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
    for v in range(1, 4):
        for i in range(3):
            base = 3 + (v-1) * 9 + i * 3
            s_map[v, i, 0], s_map[v, i, 1], s_map[v, i, 2] = base, base + 1, base + 2
            
    # Tiers 4, 5, 6, 7 (V >= 4): Trend-only (IDs 30-41)
    for v in range(4, 8):
        for i in range(3):
            base = 30 + (v-4) * 3
            s_map[v, i, 0], s_map[v, i, 1], s_map[v, i, 2] = base, base + 1, base + 2
            
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

SHARD_LABELS: List[str] = get_shard_labels(PROFILE_RGB)

# --- 4. Dispatch Kernels ---

@njit(inline='always', fastmath=True, cache=True)
def get_context_id_fast(ag: uint8, bg: uint8, cg: uint8, intensity_idx: uint8,
                        s_lut: npt.NDArray[np.uint8],
                        d_lut: npt.NDArray[np.uint8]) -> uint8:
    """ 
    Stateless 2-step LUT dispatch for shard context derivation. 
    [v8.2.1] Optimized for L2 cache residency (~300KB footprint).
    """
    # Cast to int for indexing safety in Numba
    da = int(ag) - int(cg) + 255
    db = int(bg) - int(cg) + 255
    pk = s_lut[da, db]
    return d_lut[pk, int(intensity_idx)]

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def shard_pass_1_rgb(h: int, w: int, gr_ch: npt.NDArray[np.uint8], rd_ch: npt.NDArray[np.uint8],
                     bd_ch: npt.NDArray[np.uint8], a_ch: npt.NDArray[np.uint8], is_rgba: bool,
                     n_shards: int, s_lut: npt.NDArray[np.uint8], i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint64]], Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], Tuple[np.uint64, np.float64]], npt.NDArray[np.uint8]]:
    """ 
    Stage 1: O(N) Shard Profiling with Staggered BICC Coordination.
    Optimized for L2 cache residency and branch prediction.
    """
    num_chunks = int(min(16, h)) if h > 0 else 1
    chunk_size = (h + num_chunks - 1) // num_chunks
    chunk_shard_hists = np.zeros((num_chunks, 3, n_shards, 256), dtype=np.uint32)
    row_ptrs = np.zeros((h, 3, n_shards), dtype=np.uint32)
    row_hits = np.zeros((h, 3), dtype=np.uint32)
    row_abs_sums = np.zeros((h, 3), dtype=np.uint64)
    
    gr_res = np.empty((h, w), dtype=np.uint8)
    rd_res = np.empty((h, w), dtype=np.uint8)
    bd_res = np.empty((h, w), dtype=np.uint8)
    a_res = np.empty((h, w), dtype=np.uint8) if is_rgba else np.empty((0, 0), dtype=np.uint8)
    row_a_hits = np.zeros(h, dtype=np.uint64)
    row_a_sums = np.zeros(h, dtype=np.float64)

    for c_idx in prange(num_chunks):
        start_i, end_i = c_idx * chunk_size, min((c_idx + 1) * chunk_size, h)
        local_hists = np.zeros((3, n_shards, 256), dtype=np.uint32)
        for i in range(start_i, end_i):
            pi = i + 1
            h0, h1, h2 = np.uint32(0), np.uint32(0), np.uint32(0)
            s0, s1, s2 = np.uint64(0), np.uint64(0), np.uint64(0)
            
            for j in range(w):
                pj = j + 1
                
                # 1. Green (Baseline)
                ag, bg, cg = gr_ch[pi, pj-1], gr_ch[pi-1, pj], gr_ch[pi-1, pj-1]
                pg = selected_predictor(ag, bg, cg)
                vg = gr_ch[pi, pj]
                ctxg = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
                resg_zz = ZIGZAG_LUT[np.uint8((int(vg) - int(pg)) & 0xFF)]
                
                gr_res[i, j] = resg_zz
                local_hists[0, ctxg, resg_zz] += 1
                row_ptrs[i, 0, ctxg] += 1
                h0 += np.uint32(resg_zz == 0); s0 += np.uint64(abs(int(vg) - int(pg)))
                
                # 2. Red (Dependent)
                v1 = rd_ch[pi, pj]
                a1, b1, c1 = rd_ch[pi, pj-1], rd_ch[pi-1, pj], rd_ch[pi-1, pj-1]
                idx_v = i_lut[vg] 
                ctx1 = int(get_context_id_fast(a1, b1, c1, idx_v, s_lut, d_lut))
                p1 = selected_predictor(a1, b1, c1)
                res1_zz = ZIGZAG_LUT[np.uint8((int(v1) - int(p1)) & 0xFF)]
                
                rd_res[i, j] = res1_zz
                local_hists[1, ctx1, res1_zz] += 1
                row_ptrs[i, 1, ctx1] += 1
                h1 += np.uint32(res1_zz == 0); s1 += np.uint64(abs(int(v1) - int(p1)))
                
                # 3. Blue (Dependent)
                v2 = bd_ch[pi, pj]
                a2, b2, c2 = bd_ch[pi, pj-1], bd_ch[pi-1, pj], bd_ch[pi-1, pj-1]
                ctx2 = int(get_context_id_fast(a2, b2, c2, idx_v, s_lut, d_lut))
                p2 = selected_predictor(a2, b2, c2)
                res2_zz = ZIGZAG_LUT[np.uint8((int(v2) - int(p2)) & 0xFF)]
                
                bd_res[i, j] = res2_zz
                local_hists[2, ctx2, res2_zz] += 1
                row_ptrs[i, 2, ctx2] += 1
                h2 += np.uint32(res2_zz == 0); s2 += np.uint64(abs(int(v2) - int(p2)))

            if is_rgba:
                ag_a, cg_a, h_acc, s_acc = uint8(0), uint8(0), uint64(0), 0.0
                r_s, r_ts = a_ch[i], (a_ch[i-1] if i > 0 else a_ch[i])
                for j in range(w):
                    bg_a = r_ts[j] if i > 0 else uint8(0)
                    pg_a = selected_predictor(ag_a, bg_a, cg_a)
                    r_a = ZIGZAG_LUT[np.uint8((int(r_s[j]) - int(pg_a)) & 0xFF)]
                    a_res[i, j] = r_a; s_acc += abs(float(r_s[j]) - float(pg_a)); h_acc += uint64(r_a == 0)
                    ag_a, cg_a = r_s[j], (r_ts[j] if i > 0 else uint8(0))
                row_a_hits[i], row_a_sums[i] = h_acc, s_acc
            
            row_hits[i, 0], row_hits[i, 1], row_hits[i, 2] = h0, h1, h2
            row_abs_sums[i, 0], row_abs_sums[i, 1], row_abs_sums[i, 2] = s0, s1, s2
        chunk_shard_hists[c_idx] = local_hists

    shard_stats = chunk_shard_hists.sum(axis=0)
    shard_counts = shard_stats.sum(axis=2).astype(np.uint32)
    shard_offsets = np.zeros((3, n_shards), dtype=np.uint32)
    for c in range(3):
        curr = 0
        for s in range(n_shards):
            shard_offsets[c, s] = curr
            curr += int(shard_counts[c, s])
    row_global_offsets = np.zeros((h, 3, n_shards), dtype=np.uint32)
    for c in range(3):
        for s in range(n_shards):
            curr = int(shard_offsets[c, s])
            for i in range(h):
                row_global_offsets[i, c, s] = uint32(curr)
                curr += int(row_ptrs[i, c, s])
    return shard_counts, shard_stats, shard_offsets, row_global_offsets, (row_hits.sum(axis=0), row_abs_sums.sum(axis=0)), (gr_res, rd_res, bd_res, a_res, (uint64(row_a_hits.sum()), row_a_sums.sum())), np.empty((0, 0, 0), dtype=np.uint8)


@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def shard_pass_2_rgb(h: int, w: int, ctx_map: npt.NDArray[np.uint8],
                     res_gr: npt.NDArray[np.uint8], res_rd: npt.NDArray[np.uint8], res_bd: npt.NDArray[np.uint8],
                     row_global_offsets: npt.NDArray[np.uint32],
                     shard_gr: npt.NDArray[np.uint8], shard_rd: npt.NDArray[np.uint8], shard_bd: npt.NDArray[np.uint8]):
    """ 
    Stage 2: O(N) Encoding Payload Construction.
    -------------------------------------------
    [v8.3.1] Gather-only: Uses pre-calculated Context IDs and residuals from Pass 1 
    to fill the shard-partitioned buffers. This scan is now purely a data-gathering 
    operation, as all prediction and context profiling is completed in Stage 1.
    """
    for i in prange(h):
        l_gr, l_rd, l_bd = row_global_offsets[i, 0].copy(), row_global_offsets[i, 1].copy(), row_global_offsets[i, 2].copy()
        for pj in range(w):
            ctxg = int(ctx_map[0, i, pj])
            shard_gr[l_gr[ctxg]] = res_gr[i, pj]
            l_gr[ctxg] += 1
            
            ctx1 = int(ctx_map[1, i, pj])
            ctx2 = int(ctx_map[2, i, pj])
            shard_rd[l_rd[ctx1]] = res_rd[i, pj]
            shard_bd[l_bd[ctx2]] = res_bd[i, pj]
            l_rd[ctx1] += 1; l_bd[ctx2] += 1

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def shard_pass_1_gray(h: int, w: int, gr_ch: npt.NDArray[np.uint8], a_ch: npt.NDArray[np.uint8], is_rgba: bool,
                       n_shards: int, s_lut: npt.NDArray[np.uint8], i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint64]], Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], Tuple[np.uint64, np.float64]], npt.NDArray[np.uint8]]:
    """ 
    Stage 1: O(N) Shard Profiling for Grayscale (Sequential Raster). 
    ----------------------------------------------------------
    [v8.3.1] Caching: Now fills 2D residual buffers and ctx_map to enable a pure 
    gather operation in Pass 2.
    """
    num_chunks = int(min(16, h)) if h > 0 else 1
    chunk_size = (h + num_chunks - 1) // num_chunks
    chunk_shard_hists = np.zeros((num_chunks, n_shards, 256), dtype=np.uint32)
    row_ptrs = np.zeros((h, n_shards), dtype=np.uint32)
    row_hits = np.zeros(h, dtype=np.uint32)
    row_abs_sums = np.zeros(h, dtype=np.uint64)
    gr_res = np.empty((h, w), dtype=np.uint8)
    a_res = np.empty((h, w), dtype=np.uint8) if is_rgba else np.empty((0, 0), dtype=np.uint8)
    row_a_hits = np.zeros(h, dtype=np.uint64)
    row_a_sums = np.zeros(h, dtype=np.float64)

    for c_idx in prange(num_chunks):
        start_i, end_i = c_idx * chunk_size, min((c_idx + 1) * chunk_size, h)
        local_hists = np.zeros((n_shards, 256), dtype=np.uint32)
        for i in range(start_i, end_i):
            pi = i + 1
            h_acc, s_acc = np.uint32(0), np.uint64(0)
            for pj in range(1, w + 1):
                ag, bg, cg = gr_ch[pi, pj-1], gr_ch[pi-1, pj], gr_ch[pi-1, pj-1]
                pg = selected_predictor(ag, bg, cg)
                curr_valg = gr_ch[pi, pj]
                ctxg = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
                resg_zz = ZIGZAG_LUT[np.uint8((int(curr_valg) - int(pg)) & 0xFF)]
                local_hists[ctxg, resg_zz] += 1; row_ptrs[i, ctxg] += 1
                gr_res[i, pj-1] = resg_zz
                h_acc += np.uint32(resg_zz == 0); s_acc += np.uint64(abs(int(curr_valg) - int(pg)))
            
            if is_rgba:
                ag_a, cg_a, h_acc_a, s_acc_a = uint8(0), uint8(0), uint64(0), 0.0
                r_s, r_ts = a_ch[i], (a_ch[i-1] if i > 0 else a_ch[i])
                for j in range(w):
                    bg_a = r_ts[j] if i > 0 else uint8(0)
                    pg_a = selected_predictor(ag_a, bg_a, cg_a)
                    r_a = ZIGZAG_LUT[np.uint8((int(r_s[j]) - int(pg_a)) & 0xFF)]
                    a_res[i, j] = r_a; s_acc_a += abs(float(r_s[j]) - float(pg_a)); h_acc_a += uint64(r_a == 0)
                    ag_a, cg_a = r_s[j], (r_ts[j] if i > 0 else uint8(0))
                row_a_hits[i], row_a_sums[i] = h_acc_a, s_acc_a
            
            row_hits[i], row_abs_sums[i] = h_acc, s_acc
        chunk_shard_hists[c_idx] = local_hists

    shard_stats_1ch = chunk_shard_hists.sum(axis=0)
    shard_counts_1ch = row_ptrs.sum(axis=0).astype(np.uint32)
    
    shard_offsets_1ch = np.zeros(n_shards, dtype=np.uint32)
    curr = np.uint32(0)
    for s in range(n_shards):
        shard_offsets_1ch[s] = curr
        curr += shard_counts_1ch[s]

    row_global_offsets_1ch = np.zeros((h, n_shards), dtype=np.uint32)
    for s in range(n_shards):
        curr = shard_offsets_1ch[s]
        for i in range(h):
            row_global_offsets_1ch[i, s] = curr
            curr += row_ptrs[i, s]

    # Expand to 3-channel for ShardBuffer compatibility
    shard_counts = np.zeros((3, n_shards), dtype=np.uint32); shard_counts[0] = shard_counts_1ch
    shard_stats = np.zeros((3, n_shards, 256), dtype=np.uint32); shard_stats[0] = shard_stats_1ch
    shard_offsets = np.zeros((3, n_shards), dtype=np.uint32); shard_offsets[0] = shard_offsets_1ch
    row_global_offsets = np.zeros((h, 3, n_shards), dtype=np.uint32)
    for i in range(h): row_global_offsets[i, 0] = row_global_offsets_1ch[i]
    
    hits = np.zeros(3, dtype=np.uint32); hits[0] = row_hits.sum()
    sums = np.zeros(3, dtype=np.uint64); sums[0] = row_abs_sums.sum()
    
    empty = np.empty((0, 0), dtype=np.uint8)
    return shard_counts, shard_stats, shard_offsets, row_global_offsets, (hits, sums), (gr_res, empty, empty, a_res, (uint64(row_a_hits.sum()), row_a_sums.sum())), np.empty((0, 0, 0), dtype=np.uint8)

@njit(fastmath=True, error_model='numpy', cache=True)
def shard_pass_2_gray(h: int, w: int, ctx_map: npt.NDArray[np.uint8],
                      res_gr: npt.NDArray[np.uint8],
                      row_global_offsets: npt.NDArray[np.uint32],
                      shard_gr: npt.NDArray[np.uint8]):
    """ 
    Stage 2: O(N) Encoding Payload Construction. 
    -------------------------------------------
    [v8.3.1] Gather-only: Uses pre-calculated Context IDs and residuals from Pass 1 
    to fill the shard-partitioned buffers. This scan is now purely a data-gathering 
    operation with zero-redundancy.
    """
    for i in prange(h):
        l_gr = row_global_offsets[i, 0].copy()
        for pj in range(w):
            ctxg = int(ctx_map[0, i, pj])
            shard_gr[l_gr[ctxg]] = res_gr[i, pj]
            l_gr[ctxg] += 1


@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def reconstruct_shards_rgb(h: int, w: int, res_gr: npt.NDArray[np.uint8], res_rd: npt.NDArray[np.uint8],
                           res_bd: npt.NDArray[np.uint8], off_gr: npt.NDArray[np.uint32],
                           off_rd: npt.NDArray[np.uint32], off_bd: npt.NDArray[np.uint32],
                           is_grayscale: bool, s_lut: npt.NDArray[np.uint8], 
                           i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]):
    """ 
    Stage 3: Parallel Channel Reconstruction Engine using IZIGZAG_LUT.
    ----------------------------------------------------------
    Architecture:
    1. Grn-Channel (Baseline): Reconstructs first to provide intensity context.
    2. Rd/Bd-Channel (Dependent): Reconstructs in parallel using the Grn-reconstructed 
       values as intensity indices for context derivation.
    """
    gr_rec = np.zeros((h + 2, w + 2), dtype=np.uint8)
    rd_rec = np.zeros((h + 2, w + 2), dtype=np.uint8) if not is_grayscale else np.zeros((1, 1), dtype=np.uint8)
    bd_rec = np.zeros((h + 2, w + 2), dtype=np.uint8) if not is_grayscale else np.zeros((1, 1), dtype=np.uint8)
    p_gr, p_rd, p_bd = off_gr.copy(), off_rd.copy(), off_bd.copy()
    for pi in range(1, h + 1):
        for pj in range(1, w + 1):
            ag, bg, cg = gr_rec[pi, pj-1], gr_rec[pi-1, pj], gr_rec[pi-1, pj-1]
            pg = selected_predictor(ag, bg, cg)
            ctx = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
            res = IZIGZAG_LUT[res_gr[p_gr[ctx]]]; p_gr[ctx] += 1
            gr_rec[pi, pj] = np.uint8((int(res) + int(pg)) & 0xFF)
    if not is_grayscale:
        for c_idx in prange(2):
            if c_idx == 0:
                for pi in range(1, h + 1):
                    for pj in range(1, w + 1):
                        cur_g = gr_rec[pi, pj]
                        a, b, c = rd_rec[pi, pj-1], rd_rec[pi-1, pj], rd_rec[pi-1, pj-1]
                        p = selected_predictor(a, b, c)
                        ctx = int(get_context_id_fast(a, b, c, i_lut[cur_g], s_lut, d_lut))
                        res = IZIGZAG_LUT[res_rd[p_rd[ctx]]]; p_rd[ctx] += 1
                        rd_rec[pi, pj] = np.uint8((int(res) + int(p)) & 0xFF)
            else:
                for pi in range(1, h + 1):
                    for pj in range(1, w + 1):
                        cur_g = gr_rec[pi, pj]
                        a, b, c = bd_rec[pi, pj-1], bd_rec[pi-1, pj], bd_rec[pi-1, pj-1]
                        p = selected_predictor(a, b, c)
                        ctx = int(get_context_id_fast(a, b, c, i_lut[cur_g], s_lut, d_lut))
                        res = IZIGZAG_LUT[res_bd[p_bd[ctx]]]; p_bd[ctx] += 1
                        bd_rec[pi, pj] = np.uint8((int(res) + int(p)) & 0xFF)
    res_gr = gr_rec[1:h+1, 1:w+1]
    res_rd = rd_rec[1:h+1, 1:w+1] if not is_grayscale else np.empty((0, 0), dtype=np.uint8)
    res_bd = bd_rec[1:h+1, 1:w+1] if not is_grayscale else np.empty((0, 0), dtype=np.uint8)
    return res_gr, res_rd, res_bd

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def shard_pass_2_rgb_stateless(h: int, w: int,
                                gr_ch: npt.NDArray[np.uint8], rd_ch: npt.NDArray[np.uint8], bd_ch: npt.NDArray[np.uint8],
                                row_global_offsets: npt.NDArray[np.uint32],
                                shard_gr: npt.NDArray[np.uint8], shard_rd: npt.NDArray[np.uint8], shard_bd: npt.NDArray[np.uint8],
                                s_lut: npt.NDArray[np.uint8], i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]) -> None:
    """
    Stateless Pass 2 (RGB): recomputes prediction+context per pixel from raw channels,
    eliminating the 12MB ctx_map + residual readback that the original Pass 2 required.
    """
    for i in prange(h):
        l_gr = row_global_offsets[i, 0].copy()
        l_rd = row_global_offsets[i, 1].copy()
        l_bd = row_global_offsets[i, 2].copy()
        for j in range(w):
            pj = j + 1
            pi = i + 1
            ag, bg, cg = gr_ch[pi, pj-1], gr_ch[pi-1, pj], gr_ch[pi-1, pj-1]
            pg = selected_predictor(ag, bg, cg)
            vg = gr_ch[pi, pj]
            ctxg = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
            shard_gr[l_gr[ctxg]] = ZIGZAG_LUT[np.uint8((int(vg) - int(pg)) & 0xFF)]
            l_gr[ctxg] += 1
            idx_v = i_lut[vg]
            a1, b1, c1 = rd_ch[pi, pj-1], rd_ch[pi-1, pj], rd_ch[pi-1, pj-1]
            p1 = selected_predictor(a1, b1, c1)
            ctx1 = int(get_context_id_fast(a1, b1, c1, idx_v, s_lut, d_lut))
            shard_rd[l_rd[ctx1]] = ZIGZAG_LUT[np.uint8((int(rd_ch[pi, pj]) - int(p1)) & 0xFF)]
            l_rd[ctx1] += 1
            a2, b2, c2 = bd_ch[pi, pj-1], bd_ch[pi-1, pj], bd_ch[pi-1, pj-1]
            p2 = selected_predictor(a2, b2, c2)
            ctx2 = int(get_context_id_fast(a2, b2, c2, idx_v, s_lut, d_lut))
            shard_bd[l_bd[ctx2]] = ZIGZAG_LUT[np.uint8((int(bd_ch[pi, pj]) - int(p2)) & 0xFF)]
            l_bd[ctx2] += 1


@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def shard_pass_2_gray_stateless(h: int, w: int,
                                 gr_ch: npt.NDArray[np.uint8],
                                 row_global_offsets: npt.NDArray[np.uint32],
                                 shard_gr: npt.NDArray[np.uint8],
                                 s_lut: npt.NDArray[np.uint8], i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]) -> None:
    """
    Stateless Pass 2 (Gray): recomputes prediction+context per pixel from raw channel,
    eliminating the ctx_map + residual readback that the original Pass 2 required.
    """
    for i in prange(h):
        l_gr = row_global_offsets[i, 0].copy()
        for j in range(w):
            pj = j + 1
            pi = i + 1
            ag, bg, cg = gr_ch[pi, pj-1], gr_ch[pi-1, pj], gr_ch[pi-1, pj-1]
            pg = selected_predictor(ag, bg, cg)
            vg = gr_ch[pi, pj]
            ctxg = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
            shard_gr[l_gr[ctxg]] = ZIGZAG_LUT[np.uint8((int(vg) - int(pg)) & 0xFF)]
            l_gr[ctxg] += 1


@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def fused_rct_p1_rgb(h: int, w: int, rgb_raw: npt.NDArray[np.uint8],
                      a_ch: npt.NDArray[np.uint8], is_rgba: bool,
                      n_shards: int, s_lut: npt.NDArray[np.uint8],
                      i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]):
    """
    Fused G-sub RCT + Shard Pass 1 (RGB).
    Eliminates the separate extract_channels scan and np.pad copies by writing
    directly into the padded channel arrays and accumulating channel histograms
    in the same shard-profiling pass.  Memory traffic: 24N → 18N per encode.

    Sub-pass 1 (prange by row): G-sub RCT → fills gr/rd/bd_ch_p interior; no inter-row deps.
    Sub-pass 2 (prange by chunk): MED prediction + ZigZag + shard counts + channel hists.
    Numba serializes the two prange loops so sub-pass 1 is always complete before sub-pass 2 reads.
    """
    gr_ch_p = np.zeros((h + 2, w + 2), dtype=np.uint8)
    rd_ch_p = np.zeros((h + 2, w + 2), dtype=np.uint8)
    bd_ch_p = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Sub-pass 1: G-sub RCT — parallel by row, writes padded interior
    for i in prange(h):
        pi = i + 1
        for j in range(w):
            pj = j + 1
            r = rgb_raw[i, j, 0]; g = rgb_raw[i, j, 1]; b = rgb_raw[i, j, 2]
            gr_ch_p[pi, pj] = g
            rd_ch_p[pi, pj] = uint8((int(r) - int(g)) & 0xFF)
            bd_ch_p[pi, pj] = uint8((int(b) - int(g)) & 0xFF)

    # Sub-pass 2: MED + shard profiling + channel hists — parallel by chunk
    num_chunks = int(min(16, h)) if h > 0 else 1
    chunk_size = (h + num_chunks - 1) // num_chunks
    chunk_shard_hists = np.zeros((num_chunks, 3, n_shards, 256), dtype=np.uint32)
    chunk_ch_hists = np.zeros((num_chunks, 3, 256), dtype=np.uint32)
    row_ptrs = np.zeros((h, 3, n_shards), dtype=np.uint32)
    row_hits = np.zeros((h, 3), dtype=np.uint32)
    row_abs_sums = np.zeros((h, 3), dtype=np.uint64)
    gr_res = np.empty((h, w), dtype=np.uint8)
    rd_res = np.empty((h, w), dtype=np.uint8)
    bd_res = np.empty((h, w), dtype=np.uint8)
    a_res = np.empty((h, w), dtype=np.uint8) if is_rgba else np.empty((0, 0), dtype=np.uint8)
    row_a_hits = np.zeros(h, dtype=np.uint64)
    row_a_sums = np.zeros(h, dtype=np.float64)

    for c_idx in prange(num_chunks):
        start_i = c_idx * chunk_size
        end_i = min((c_idx + 1) * chunk_size, h)
        local_hists = np.zeros((3, n_shards, 256), dtype=np.uint32)
        local_ch = np.zeros((3, 256), dtype=np.uint32)

        for i in range(start_i, end_i):
            pi = i + 1
            h0, h1, h2 = np.uint32(0), np.uint32(0), np.uint32(0)
            s0, s1, s2 = np.uint64(0), np.uint64(0), np.uint64(0)

            for j in range(w):
                pj = j + 1
                vg = gr_ch_p[pi, pj]
                ag, bg, cg = gr_ch_p[pi, pj-1], gr_ch_p[pi-1, pj], gr_ch_p[pi-1, pj-1]
                pg = selected_predictor(ag, bg, cg)
                ctxg = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
                resg_zz = ZIGZAG_LUT[uint8((int(vg) - int(pg)) & 0xFF)]
                gr_res[i, j] = resg_zz
                local_hists[0, ctxg, resg_zz] += 1
                row_ptrs[i, 0, ctxg] += 1
                h0 += np.uint32(resg_zz == 0); s0 += np.uint64(abs(int(vg) - int(pg)))
                local_ch[0, vg] += 1

                v1 = rd_ch_p[pi, pj]
                a1, b1, c1 = rd_ch_p[pi, pj-1], rd_ch_p[pi-1, pj], rd_ch_p[pi-1, pj-1]
                idx_v = i_lut[vg]
                p1 = selected_predictor(a1, b1, c1)
                ctx1 = int(get_context_id_fast(a1, b1, c1, idx_v, s_lut, d_lut))
                res1_zz = ZIGZAG_LUT[uint8((int(v1) - int(p1)) & 0xFF)]
                rd_res[i, j] = res1_zz
                local_hists[1, ctx1, res1_zz] += 1
                row_ptrs[i, 1, ctx1] += 1
                h1 += np.uint32(res1_zz == 0); s1 += np.uint64(abs(int(v1) - int(p1)))
                local_ch[1, v1] += 1

                v2 = bd_ch_p[pi, pj]
                a2, b2, c2 = bd_ch_p[pi, pj-1], bd_ch_p[pi-1, pj], bd_ch_p[pi-1, pj-1]
                p2 = selected_predictor(a2, b2, c2)
                ctx2 = int(get_context_id_fast(a2, b2, c2, idx_v, s_lut, d_lut))
                res2_zz = ZIGZAG_LUT[uint8((int(v2) - int(p2)) & 0xFF)]
                bd_res[i, j] = res2_zz
                local_hists[2, ctx2, res2_zz] += 1
                row_ptrs[i, 2, ctx2] += 1
                h2 += np.uint32(res2_zz == 0); s2 += np.uint64(abs(int(v2) - int(p2)))
                local_ch[2, v2] += 1

            if is_rgba:
                ag_a, cg_a, h_acc, s_acc = uint8(0), uint8(0), uint64(0), 0.0
                r_s = a_ch[i]
                r_ts = a_ch[i-1] if i > 0 else a_ch[i]
                for j in range(w):
                    bg_a = r_ts[j] if i > 0 else uint8(0)
                    pg_a = selected_predictor(ag_a, bg_a, cg_a)
                    r_a = ZIGZAG_LUT[uint8((int(r_s[j]) - int(pg_a)) & 0xFF)]
                    a_res[i, j] = r_a
                    s_acc += abs(float(r_s[j]) - float(pg_a))
                    h_acc += uint64(r_a == 0)
                    ag_a = r_s[j]
                    cg_a = r_ts[j] if i > 0 else uint8(0)
                row_a_hits[i] = h_acc
                row_a_sums[i] = s_acc

            row_hits[i, 0] = h0; row_hits[i, 1] = h1; row_hits[i, 2] = h2
            row_abs_sums[i, 0] = s0; row_abs_sums[i, 1] = s1; row_abs_sums[i, 2] = s2

        chunk_shard_hists[c_idx] = local_hists
        chunk_ch_hists[c_idx] = local_ch

    shard_stats = chunk_shard_hists.sum(axis=0)
    channel_hists = chunk_ch_hists.sum(axis=0)
    shard_counts = shard_stats.sum(axis=2).astype(np.uint32)

    shard_offsets = np.zeros((3, n_shards), dtype=np.uint32)
    for c in range(3):
        curr = 0
        for s in range(n_shards):
            shard_offsets[c, s] = curr
            curr += int(shard_counts[c, s])

    row_global_offsets = np.zeros((h, 3, n_shards), dtype=np.uint32)
    for c in range(3):
        for s in range(n_shards):
            curr = int(shard_offsets[c, s])
            for i in range(h):
                row_global_offsets[i, c, s] = uint32(curr)
                curr += int(row_ptrs[i, c, s])

    hits = row_hits.sum(axis=0)
    sums = row_abs_sums.sum(axis=0)
    res_cached = (gr_res, rd_res, bd_res, a_res, (uint64(row_a_hits.sum()), row_a_sums.sum()))

    return (gr_ch_p, rd_ch_p, bd_ch_p, channel_hists,
            shard_counts, shard_stats, shard_offsets, row_global_offsets,
            (hits, sums), res_cached)


@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def fused_rct_p1_gray(h: int, w: int, gray_raw: npt.NDArray[np.uint8],
                       a_ch: npt.NDArray[np.uint8], is_rgba: bool,
                       n_shards: int, s_lut: npt.NDArray[np.uint8],
                       i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]):
    """
    Fused pad + Shard Pass 1 (Grayscale).
    Writes directly into the padded channel array, avoiding the separate np.pad copy.
    """
    gr_ch_p = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Sub-pass 1: copy into padded interior — parallel by row
    for i in prange(h):
        pi = i + 1
        for j in range(w):
            gr_ch_p[pi, j + 1] = gray_raw[i, j]

    # Sub-pass 2: MED + shard profiling — parallel by chunk (same as shard_pass_1_gray body)
    num_chunks = int(min(16, h)) if h > 0 else 1
    chunk_size = (h + num_chunks - 1) // num_chunks
    chunk_shard_hists = np.zeros((num_chunks, n_shards, 256), dtype=np.uint32)
    row_ptrs = np.zeros((h, n_shards), dtype=np.uint32)
    row_hits = np.zeros(h, dtype=np.uint32)
    row_abs_sums = np.zeros(h, dtype=np.uint64)
    gr_res = np.empty((h, w), dtype=np.uint8)
    a_res = np.empty((h, w), dtype=np.uint8) if is_rgba else np.empty((0, 0), dtype=np.uint8)
    row_a_hits = np.zeros(h, dtype=np.uint64)
    row_a_sums = np.zeros(h, dtype=np.float64)

    for c_idx in prange(num_chunks):
        start_i = c_idx * chunk_size
        end_i = min((c_idx + 1) * chunk_size, h)
        local_hists = np.zeros((n_shards, 256), dtype=np.uint32)

        for i in range(start_i, end_i):
            pi = i + 1
            h_acc, s_acc = np.uint32(0), np.uint64(0)
            for pj in range(1, w + 1):
                ag, bg, cg = gr_ch_p[pi, pj-1], gr_ch_p[pi-1, pj], gr_ch_p[pi-1, pj-1]
                pg = selected_predictor(ag, bg, cg)
                curr_valg = gr_ch_p[pi, pj]
                ctxg = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
                resg_zz = ZIGZAG_LUT[uint8((int(curr_valg) - int(pg)) & 0xFF)]
                local_hists[ctxg, resg_zz] += 1
                row_ptrs[i, ctxg] += 1
                gr_res[i, pj-1] = resg_zz
                h_acc += np.uint32(resg_zz == 0)
                s_acc += np.uint64(abs(int(curr_valg) - int(pg)))

            if is_rgba:
                ag_a, cg_a, h_acc_a, s_acc_a = uint8(0), uint8(0), uint64(0), 0.0
                r_s = a_ch[i]
                r_ts = a_ch[i-1] if i > 0 else a_ch[i]
                for j in range(w):
                    bg_a = r_ts[j] if i > 0 else uint8(0)
                    pg_a = selected_predictor(ag_a, bg_a, cg_a)
                    r_a = ZIGZAG_LUT[uint8((int(r_s[j]) - int(pg_a)) & 0xFF)]
                    a_res[i, j] = r_a
                    s_acc_a += abs(float(r_s[j]) - float(pg_a))
                    h_acc_a += uint64(r_a == 0)
                    ag_a = r_s[j]
                    cg_a = r_ts[j] if i > 0 else uint8(0)
                row_a_hits[i] = h_acc_a
                row_a_sums[i] = s_acc_a

            row_hits[i] = h_acc
            row_abs_sums[i] = s_acc
        chunk_shard_hists[c_idx] = local_hists

    shard_stats_1ch = chunk_shard_hists.sum(axis=0)
    shard_counts_1ch = row_ptrs.sum(axis=0).astype(np.uint32)

    shard_offsets_1ch = np.zeros(n_shards, dtype=np.uint32)
    curr = np.uint32(0)
    for s in range(n_shards):
        shard_offsets_1ch[s] = curr
        curr += shard_counts_1ch[s]

    row_global_offsets_1ch = np.zeros((h, n_shards), dtype=np.uint32)
    for s in range(n_shards):
        curr = shard_offsets_1ch[s]
        for i in range(h):
            row_global_offsets_1ch[i, s] = curr
            curr += row_ptrs[i, s]

    shard_counts = np.zeros((3, n_shards), dtype=np.uint32); shard_counts[0] = shard_counts_1ch
    shard_stats = np.zeros((3, n_shards, 256), dtype=np.uint32); shard_stats[0] = shard_stats_1ch
    shard_offsets = np.zeros((3, n_shards), dtype=np.uint32); shard_offsets[0] = shard_offsets_1ch
    row_global_offsets = np.zeros((h, 3, n_shards), dtype=np.uint32)
    for i in range(h): row_global_offsets[i, 0] = row_global_offsets_1ch[i]

    hits = np.zeros(3, dtype=np.uint32); hits[0] = row_hits.sum()
    sums = np.zeros(3, dtype=np.uint64); sums[0] = row_abs_sums.sum()
    empty = np.empty((0, 0), dtype=np.uint8)
    res_cached = (gr_res, empty, empty, a_res, (uint64(row_a_hits.sum()), row_a_sums.sum()))

    return (gr_ch_p, np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8),
            np.zeros((3, 256), dtype=np.uint32),
            shard_counts, shard_stats, shard_offsets, row_global_offsets,
            (hits, sums), res_cached)


def execute_sharding(h: int, w: int, gr_ch: npt.NDArray[np.uint8], rd_ch: npt.NDArray[np.uint8],
                     bd_ch: npt.NDArray[np.uint8], a_ch: npt.NDArray[np.uint8], 
                     is_rgba: bool, is_grayscale: bool, profile: ShardProfile,
                     p1_cached: Optional[Tuple] = None) -> ShardBuffer:
    """ 
    Unified Sharding Orchestrator (Pillar 4 Hub).
    -------------------------------------------
    Orchestrates the 2-pass sharding pipeline:
    1. Pass 1 (Profiling): Calculates shard histograms and row offsets.
       [v8.3.1] Also generates 2D residual and context maps for caching.
    2. Pass 2 (Payload): Fills shard buffers using gathered residuals.
    
    If `p1_cached` is provided, it skips Pass 1 and uses the cached results, 
    maximizing throughput for paths like Bitplane selection.
    """
    # Unpack profile for Numba compatibility
    n_shards = profile.total_shards
    s_lut, i_lut, d_lut = profile.spatial_lut, profile.intensity_lut, profile.dispatch_lut

    if p1_cached is not None:
        counts, stats, offsets, row_offs, metrics, res_cached, ctx_map = p1_cached
    else:
        if is_grayscale:
            counts, stats, offsets, row_offs, metrics, res_cached, ctx_map = shard_pass_1_gray(h, w, gr_ch, a_ch, is_rgba, n_shards, s_lut, i_lut, d_lut)
        else:
            counts, stats, offsets, row_offs, metrics, res_cached, ctx_map = shard_pass_1_rgb(h, w, gr_ch, rd_ch, bd_ch, a_ch, is_rgba, n_shards, s_lut, i_lut, d_lut)

    gr_res, rd_res, bd_res, a_res, a_metrics = res_cached
    
    if is_grayscale:
        s_gr = np.zeros(int(counts[0].sum()), dtype=np.uint8)
        shard_pass_2_gray_stateless(h, w, gr_ch, row_offs, s_gr, s_lut, i_lut, d_lut)
        s_rd = np.empty(0, dtype=np.uint8)
        s_bd = np.empty(0, dtype=np.uint8)
    else:
        s_gr = np.zeros(int(counts[0].sum()), dtype=np.uint8)
        s_rd = np.zeros(int(counts[1].sum()), dtype=np.uint8)
        s_bd = np.zeros(int(counts[2].sum()), dtype=np.uint8)
        shard_pass_2_rgb_stateless(h, w, gr_ch, rd_ch, bd_ch, row_offs, s_gr, s_rd, s_bd, s_lut, i_lut, d_lut)
    
    return ShardBuffer(s_gr, s_rd, s_bd, a_res, counts, stats, offsets, row_offs, metrics[0], metrics[1], a_metrics)
