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

__version__ = "8.3.2"

import numpy as np
import numpy.typing as npt
from numba import njit, uint8, prange, uint32, uint16, uint64
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .predictor import (ZIGZAG_LUT, IZIGZAG_LUT, selected_predictor)


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
    """ Determines the observed ZigZag symbol width per shard for PDF compaction. """
    n_channels, n_shards = shard_stats.shape[0], shard_stats.shape[1]
    widths = np.ones((n_channels, n_shards), dtype=np.uint16)
    for c in range(n_channels):
        for s in range(n_shards):
            hist = shard_stats[c, s]
            indices = np.where(hist > 0)[0]
            if indices.size > 0:
                widths[c, s] = np.uint16(int(indices[-1]) + 1)
    return widths

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
    """ Generates profile-specific LUTs for context features. """
    i_lut = np.zeros(256, dtype=np.uint8)
    i_arr = np.arange(256, dtype=np.uint8)
    for thr in i_segs[1:-1]:
        i_lut += (i_arr > int(thr)).astype(np.uint8)

    s_lut = np.zeros((511, 511), dtype=np.uint8)
    d = np.arange(-255, 256, dtype=np.int16)
    DA, DB = np.meshgrid(d, d, indexing='ij')

    v_mag = np.maximum(np.abs(DA), np.abs(DB))
    v_tier = (v_mag > 0).astype(np.uint8)
    for i in range(1, len(v_bounds) - 1):
        v_tier += (v_mag > int(v_bounds[i])).astype(np.uint8)

    rising  = ((DA > 0) & (DB > 0)).astype(np.uint8)
    falling = ((DA < 0) & (DB < 0)).astype(np.uint8)
    t_idx   = (falling + 2 * (1 - rising - falling)).astype(np.uint8)

    ns_hit = ((np.abs(DA) > 12) & (np.abs(DB) > 12)).astype(np.uint8)
    s_lut[:] = ((v_tier << 3) | (t_idx << 1) | ns_hit).astype(np.uint8)

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

# --- 4. Dispatch Kernels ---

@njit(inline='always', fastmath=True, cache=True)
def get_context_id_fast(ag: uint8, bg: uint8, cg: uint8, intensity_idx: uint8,
                        s_lut: npt.NDArray[np.uint8],
                        d_lut: npt.NDArray[np.uint8]) -> uint8:
    da = int(ag) - int(cg) + 255
    db = int(bg) - int(cg) + 255
    pk = s_lut[da, db]
    return d_lut[pk, int(intensity_idx)]

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def fused_rct_p1_rgb(h: int, w: int, rgb_raw: npt.NDArray[np.uint8],
                      a_ch: npt.NDArray[np.uint8], is_rgba: bool,
                      n_shards: int, s_lut: npt.NDArray[np.uint8],
                      i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint64], Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]], Tuple[np.uint64, np.float64]]:
    """
    Fused Architecture: Pillar 4 Core.
    1. RCT Transform: Decorrelates RGB -> G, R-G, B-G.
    2. Pass 1 Profiling: Calculates BICC statistics and shard pointers.
    3. Spatial Caching: Generates 2D residuals for high-entropy fallback (Bitplane).
    
    This fused approach reduces DRAM traffic by 3x compared to sequential passes.
    """
    gr_ch_p = np.zeros((h + 2, w + 2), dtype=np.uint8)
    rd_ch_p = np.zeros((h + 2, w + 2), dtype=np.uint8)
    bd_ch_p = np.zeros((h + 2, w + 2), dtype=np.uint8)

    for i in prange(h):
        pi = i + 1
        for j in range(w):
            pj = j + 1
            r, g, b = rgb_raw[i, j, 0], rgb_raw[i, j, 1], rgb_raw[i, j, 2]
            gr_ch_p[pi, pj] = g
            rd_ch_p[pi, pj] = uint8((int(r) - int(g)) & 0xFF)
            bd_ch_p[pi, pj] = uint8((int(b) - int(g)) & 0xFF)

    num_chunks = int(min(16, h)) if h > 0 else 1
    chunk_size = (h + num_chunks - 1) // num_chunks
    chunk_shard_hists = np.zeros((num_chunks, 3, n_shards, 256), dtype=np.uint32)
    chunk_ch_hists = np.zeros((num_chunks, 3, 256), dtype=np.uint32)
    row_ptrs = np.zeros((h, 3, n_shards), dtype=np.uint32)
    row_hits = np.zeros((h, 3), dtype=np.uint32)
    row_abs_sums = np.zeros((h, 3), dtype=np.uint64)
    gr_res = np.zeros((h + 2, w + 2), dtype=np.uint8)
    rd_res = np.zeros((h + 2, w + 2), dtype=np.uint8)
    bd_res = np.zeros((h + 2, w + 2), dtype=np.uint8)
    a_res = np.empty((h, w), dtype=np.uint8) if is_rgba else np.empty((0, 0), dtype=np.uint8)
    row_a_hits = np.zeros(h, dtype=np.uint64)
    row_a_sums = np.zeros(h, dtype=np.float64)

    for c_idx in prange(num_chunks):
        start_i, end_i = c_idx * chunk_size, min((c_idx + 1) * chunk_size, h)
        local_hists, local_ch = np.zeros((3, n_shards, 256), dtype=np.uint32), np.zeros((3, 256), dtype=np.uint32)
        for i in range(start_i, end_i):
            pi = i + 1
            h0, h1, h2, s0, s1, s2 = np.uint32(0), np.uint32(0), np.uint32(0), np.uint64(0), np.uint64(0), np.uint64(0)
            for j in range(w):
                pj = j + 1
                vg = gr_ch_p[pi, pj]
                ag, bg, cg = gr_ch_p[pi, pj-1], gr_ch_p[pi-1, pj], gr_ch_p[pi-1, pj-1]
                pg = selected_predictor(ag, bg, cg)
                ctxg = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
                resg_zz = ZIGZAG_LUT[uint8((int(vg) - int(pg)) & 0xFF)]
                gr_res[pi, pj], row_ptrs[i, 0, ctxg] = resg_zz, row_ptrs[i, 0, ctxg] + 1
                local_hists[0, ctxg, resg_zz] += 1; local_ch[0, vg] += 1
                h0 += np.uint32(resg_zz == 0); s0 += np.uint64(abs(int(vg) - int(pg)))

                v1 = rd_ch_p[pi, pj]
                a1, b1, c1 = rd_ch_p[pi, pj-1], rd_ch_p[pi-1, pj], rd_ch_p[pi-1, pj-1]
                idx_v, p1 = i_lut[vg], selected_predictor(a1, b1, c1)
                ctx1 = int(get_context_id_fast(a1, b1, c1, idx_v, s_lut, d_lut))
                res1_zz = ZIGZAG_LUT[uint8((int(v1) - int(p1)) & 0xFF)]
                rd_res[pi, pj], row_ptrs[i, 1, ctx1] = res1_zz, row_ptrs[i, 1, ctx1] + 1
                local_hists[1, ctx1, res1_zz] += 1; local_ch[1, v1] += 1
                h1 += np.uint32(res1_zz == 0); s1 += np.uint64(abs(int(v1) - int(p1)))

                v2 = bd_ch_p[pi, pj]
                a2, b2, c2 = bd_ch_p[pi, pj-1], bd_ch_p[pi-1, pj], bd_ch_p[pi-1, pj-1]
                p2 = selected_predictor(a2, b2, c2)
                ctx2 = int(get_context_id_fast(a2, b2, c2, idx_v, s_lut, d_lut))
                res2_zz = ZIGZAG_LUT[uint8((int(v2) - int(p2)) & 0xFF)]
                bd_res[pi, pj], row_ptrs[i, 2, ctx2] = res2_zz, row_ptrs[i, 2, ctx2] + 1
                local_hists[2, ctx2, res2_zz] += 1; local_ch[2, v2] += 1
                h2 += np.uint32(res2_zz == 0); s2 += np.uint64(abs(int(v2) - int(p2)))

            if is_rgba:
                ag_a, cg_a, h_acc, s_acc = uint8(0), uint8(0), uint64(0), 0.0
                r_s, r_ts = a_ch[i], (a_ch[i-1] if i > 0 else a_ch[i])
                for j in range(w):
                    bg_a = r_ts[j] if i > 0 else uint8(0)
                    pg_a = selected_predictor(ag_a, bg_a, cg_a)
                    r_a = ZIGZAG_LUT[uint8((int(r_s[j]) - int(pg_a)) & 0xFF)]
                    a_res[i, j] = r_a; s_acc += abs(float(r_s[j]) - float(pg_a)); h_acc += uint64(r_a == 0)
                    ag_a, cg_a = r_s[j], (r_ts[j] if i > 0 else uint8(0))
                row_a_hits[i], row_a_sums[i] = h_acc, s_acc
            row_hits[i, 0], row_hits[i, 1], row_hits[i, 2] = h0, h1, h2
            row_abs_sums[i, 0], row_abs_sums[i, 1], row_abs_sums[i, 2] = s0, s1, s2
        chunk_shard_hists[c_idx], chunk_ch_hists[c_idx] = local_hists, local_ch

    shard_stats, channel_hists = chunk_shard_hists.sum(axis=0), chunk_ch_hists.sum(axis=0)
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
    res_cached = (gr_res, rd_res, bd_res, a_res)
    a_metrics = (uint64(row_a_hits.sum()), row_a_sums.sum())
    return (gr_ch_p, rd_ch_p, bd_ch_p, channel_hists, shard_counts, shard_stats, shard_offsets, row_global_offsets, row_hits.sum(axis=0), row_abs_sums.sum(axis=0), res_cached, a_metrics)

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def fused_rct_p1_gray(h: int, w: int, gray_raw: npt.NDArray[np.uint8],
                       a_ch: npt.NDArray[np.uint8], is_rgba: bool,
                       n_shards: int, s_lut: npt.NDArray[np.uint8],
                       i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint64], Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]], Tuple[np.uint64, np.float64]]:
    gr_ch_p = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for i in prange(h):
        for j in range(w): gr_ch_p[i+1, j+1] = gray_raw[i, j]

    num_chunks = int(min(16, h)) if h > 0 else 1
    chunk_size = (h + num_chunks - 1) // num_chunks
    chunk_shard_hists = np.zeros((num_chunks, n_shards, 256), dtype=np.uint32)
    row_ptrs, row_hits, row_abs_sums = np.zeros((h, n_shards), dtype=np.uint32), np.zeros(h, dtype=np.uint32), np.zeros(h, dtype=np.uint64)
    gr_res = np.zeros((h + 2, w + 2), dtype=np.uint8)
    a_res = np.empty((h, w), dtype=np.uint8) if is_rgba else np.empty((0, 0), dtype=np.uint8)
    row_a_hits, row_a_sums = np.zeros(h, dtype=np.uint64), np.zeros(h, dtype=np.float64)

    for c_idx in prange(num_chunks):
        start_i, end_i = c_idx * chunk_size, min((c_idx + 1) * chunk_size, h)
        local_hists = np.zeros((n_shards, 256), dtype=np.uint32)
        for i in range(start_i, end_i):
            pi, h_acc, s_acc = i + 1, np.uint32(0), np.uint64(0)
            for pj in range(1, w + 1):
                ag, bg, cg = gr_ch_p[pi, pj-1], gr_ch_p[pi-1, pj], gr_ch_p[pi-1, pj-1]
                pg, curr_valg = selected_predictor(ag, bg, cg), gr_ch_p[pi, pj]
                ctxg = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
                resg_zz = ZIGZAG_LUT[uint8((int(curr_valg) - int(pg)) & 0xFF)]
                local_hists[ctxg, resg_zz] += 1; row_ptrs[i, ctxg] += 1; gr_res[pi, pj] = resg_zz
                h_acc += np.uint32(resg_zz == 0); s_acc += np.uint64(abs(int(curr_valg) - int(pg)))
            if is_rgba:
                ag_a, cg_a, h_acc_a, s_acc_a = uint8(0), uint8(0), uint64(0), 0.0
                r_s, r_ts = a_ch[i], (a_ch[i-1] if i > 0 else a_ch[i])
                for j in range(w):
                    bg_a = r_ts[j] if i > 0 else uint8(0)
                    pg_a = selected_predictor(ag_a, bg_a, cg_a)
                    r_a = ZIGZAG_LUT[uint8((int(r_s[j]) - int(pg_a)) & 0xFF)]
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

    shard_counts, shard_stats, shard_offsets = np.zeros((3, n_shards), dtype=np.uint32), np.zeros((3, n_shards, 256), dtype=np.uint32), np.zeros((3, n_shards), dtype=np.uint32)
    shard_counts[0], shard_stats[0], shard_offsets[0] = shard_counts_1ch, shard_stats_1ch, shard_offsets_1ch
    row_global_offsets = np.zeros((h, 3, n_shards), dtype=np.uint32)
    for i in range(h): row_global_offsets[i, 0] = row_global_offsets_1ch[i]
    hits, sums = np.zeros(3, dtype=np.uint32), np.zeros(3, dtype=np.uint64)
    hits[0], sums[0] = row_hits.sum(), row_abs_sums.sum()
    empty_pad = np.empty((0, 0), dtype=np.uint8)
    res_cached = (gr_res, empty_pad, empty_pad, a_res)
    return (gr_ch_p, shard_counts, shard_stats, shard_offsets, row_global_offsets, hits, sums, res_cached, (uint64(row_a_hits.sum()), row_a_sums.sum()))

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def shard_pass_2_rgb_stateless(h: int, w: int,
                                gr_ch: npt.NDArray[np.uint8], rd_ch: npt.NDArray[np.uint8], bd_ch: npt.NDArray[np.uint8],
                                row_global_offsets: npt.NDArray[np.uint32],
                                shard_gr: npt.NDArray[np.uint8], shard_rd: npt.NDArray[np.uint8], shard_bd: npt.NDArray[np.uint8],
                                s_lut: npt.NDArray[np.uint8], i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]) -> None:
    for i in prange(h):
        l_gr, l_rd, l_bd = row_global_offsets[i, 0].copy(), row_global_offsets[i, 1].copy(), row_global_offsets[i, 2].copy()
        for j in range(w):
            pj, pi = j + 1, i + 1
            ag, bg, cg = gr_ch[pi, pj-1], gr_ch[pi-1, pj], gr_ch[pi-1, pj-1]
            pg, vg = selected_predictor(ag, bg, cg), gr_ch[pi, pj]
            ctxg = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
            shard_gr[l_gr[ctxg]], idx_v = ZIGZAG_LUT[uint8((int(vg) - int(pg)) & 0xFF)], i_lut[vg]
            l_gr[ctxg] += 1
            a1, b1, c1 = rd_ch[pi, pj-1], rd_ch[pi-1, pj], rd_ch[pi-1, pj-1]
            p1 = selected_predictor(a1, b1, c1)
            ctx1 = int(get_context_id_fast(a1, b1, c1, idx_v, s_lut, d_lut))
            shard_rd[l_rd[ctx1]] = ZIGZAG_LUT[uint8((int(rd_ch[pi, pj]) - int(p1)) & 0xFF)]
            l_rd[ctx1] += 1
            a2, b2, c2 = bd_ch[pi, pj-1], bd_ch[pi-1, pj], bd_ch[pi-1, pj-1]
            p2 = selected_predictor(a2, b2, c2)
            ctx2 = int(get_context_id_fast(a2, b2, c2, idx_v, s_lut, d_lut))
            shard_bd[l_bd[ctx2]] = ZIGZAG_LUT[uint8((int(bd_ch[pi, pj]) - int(p2)) & 0xFF)]
            l_bd[ctx2] += 1

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def shard_pass_2_gray_stateless(h: int, w: int,
                                 gr_ch: npt.NDArray[np.uint8],
                                 row_global_offsets: npt.NDArray[np.uint32],
                                 shard_gr: npt.NDArray[np.uint8],
                                 s_lut: npt.NDArray[np.uint8], i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]) -> None:
    for i in prange(h):
        l_gr = row_global_offsets[i, 0].copy()
        for j in range(w):
            pj, pi = j + 1, i + 1
            ag, bg, cg = gr_ch[pi, pj-1], gr_ch[pi-1, pj], gr_ch[pi-1, pj-1]
            pg, vg = selected_predictor(ag, bg, cg), gr_ch[pi, pj]
            ctxg = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
            shard_gr[l_gr[ctxg]] = ZIGZAG_LUT[uint8((int(vg) - int(pg)) & 0xFF)]
            l_gr[ctxg] += 1

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def reconstruct_shards_rgb(h: int, w: int, res_gr: npt.NDArray[np.uint8], res_rd: npt.NDArray[np.uint8],
                           res_bd: npt.NDArray[np.uint8], off_gr: npt.NDArray[np.uint32],
                           off_rd: npt.NDArray[np.uint32], off_bd: npt.NDArray[np.uint32],
                           s_lut: npt.NDArray[np.uint8],
                           i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]):
    gr_rec = np.zeros((h + 2, w + 2), dtype=np.uint8)
    rd_rec = np.zeros((h + 2, w + 2), dtype=np.uint8)
    bd_rec = np.zeros((h + 2, w + 2), dtype=np.uint8)
    p_gr, p_rd, p_bd = off_gr.copy(), off_rd.copy(), off_bd.copy()
    for pi in range(1, h + 1):
        for pj in range(1, w + 1):
            ag, bg, cg = gr_rec[pi, pj-1], gr_rec[pi-1, pj], gr_rec[pi-1, pj-1]
            pg = selected_predictor(ag, bg, cg)
            ctx = int(get_context_id_fast(ag, bg, cg, i_lut[pg], s_lut, d_lut))
            res = IZIGZAG_LUT[res_gr[p_gr[ctx]]]; p_gr[ctx] += 1
            gr_rec[pi, pj] = np.uint8((int(res) + int(pg)) & 0xFF)
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
    return gr_rec[1:h+1, 1:w+1], rd_rec[1:h+1, 1:w+1], bd_rec[1:h+1, 1:w+1]

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
