"""
ZPNG-CSDE v7.2 [High-Performance Parallel Architecture]
Module: zpng_shard_rgb
Role: Pillar 3.5 - Data Partitioning (BICC).
Description: Unified sharding orchestration compatible with 3D Mapping LUTs.

Logic Path Visualization:
```mermaid
graph TD
    A[Raw Channels: G, RD, BD] --> B[predict_pass_1: Profile Shards]
    B --> C{BICC Staggering}
    C -->|Lead| D[Green MED + Row Hist]
    C -->|Lag| E[RD/BD MED + Staggered G Context]
    D & E --> F[Global Histogram Aggregation]
    F --> G[predict_pass_2: Payload Assembly]
    G --> H[Alpha Re-centering]
    H --> I[Flat Payload Buffers]
```
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange, uint8, uint16, uint32, uint64
from typing import Tuple, List, Optional
from .common import (
    to_zigzag, from_zigzag, predict_med_standard,
    get_context_id_fast
)

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def predict_pass_1(h: int, w: int, gr_ch: npt.NDArray[np.uint8], rd_ch: npt.NDArray[np.uint8],
                   bd_ch: npt.NDArray[np.uint8], is_grayscale: bool,
                   shard_map: npt.NDArray[np.uint8], v_bounds: npt.NDArray[np.uint8],
                   i_segs: npt.NDArray[np.uint8], nsid: int) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint64]]]:
    """
    Stage 1: O(N) Shard Profiling & Histogram Generation.
    
    Architecture (Staggered Look-ahead):
    Extracts predictive residuals natively using MED (Median Edge Detection).
    Uses a 1-pixel staggered shift between Green (Lead) and RD/BD (Lag) channels: G processes
    pixel j while RD/BD process pixel j-1, using the same-position G value as context.
    This interleaved order enables pipeline parallelism while keeping context consistent
    between encoder (prev_valg) and decoder (gr_rec[i, j]) for the same target pixel.
    
    Produces detailed histograms for each 'Context Shard' so the selector can compile
    specialized PDF templates. The outputs here directly shape the bitstream container.
    """
    n_shards = shard_map.max() + 1 if nsid < 0 else nsid + 1
    num_chunks = int(min(16, h)) if h > 0 else 1
    chunk_size = (h + num_chunks - 1) // num_chunks
    chunk_shard_hists = np.zeros((num_chunks, 3, n_shards, 256), dtype=np.uint32)

    row_ptrs: npt.NDArray[np.uint32] = np.zeros((h, 3, n_shards), dtype=np.uint32)
    row_hits: npt.NDArray[np.uint32] = np.zeros((h, 3), dtype=np.uint32)
    row_abs_sums: npt.NDArray[np.uint64] = np.zeros((h, 3), dtype=np.uint64)

    for c_idx in prange(num_chunks):
        start_i = c_idx * chunk_size
        end_i = min(start_i + chunk_size, h)
        local_hists = np.zeros((3, n_shards, 256), dtype=np.uint32)

        for i in range(start_i, end_i):
            h0, h1, h2 = np.uint32(0), np.uint32(0), np.uint32(0)
            s0, s1, s2 = np.uint64(0), np.uint64(0), np.uint64(0)
            
            curr_valg = np.uint8(0)
            prev_valg = np.uint8(0)
            
            # [Unrolled Start] Pixel j=0: G-Lead Only
            if w > 0:
                ag = np.uint8(0)
                bg = (gr_ch[i-1, 0] if i > 0 else np.uint8(0))
                cg = np.uint8(0)
                
                pg = predict_med_standard(ag, bg, cg)
                curr_valg = gr_ch[i, 0]
                ctxg = int(get_context_id_fast(ag, bg, cg, pg, shard_map, nsid))
                diffg = (int(curr_valg) - int(pg)) & 0xFF
                # [v6.5] Centered storage for linear median calculation (0 -> 128)
                resg_c = (diffg + 128) & 0xFF
                local_hists[0, int(ctxg), resg_c] += 1; row_ptrs[i, 0, int(ctxg)] += 1
                h_val = resg_c - 128
                h0 += np.uint32(h_val == 0); s0 += np.uint64(abs(h_val))
                prev_valg = curr_valg

            # [Main Loop] j=1 to w-1: Dual-Channel Operation (No Stagger Branching)
            for j in range(1, w):
                # 1. G-Channel (Lead)
                ag = gr_ch[i, j-1]
                bg = (gr_ch[i-1, j] if i > 0 else np.uint8(0))
                cg = (gr_ch[i-1, j-1] if i > 0 else np.uint8(0))
                pg = predict_med_standard(ag, bg, cg)
                curr_valg = gr_ch[i, j]
                ctxg = int(get_context_id_fast(ag, bg, cg, pg, shard_map, nsid))
                diffg = (int(curr_valg) - int(pg)) & 0xFF
                # [v6.5] Centered storage for linear median calculation (0 -> 128)
                resg_c = (diffg + 128) & 0xFF
                local_hists[0, int(ctxg), resg_c] += 1; row_ptrs[i, 0, int(ctxg)] += 1
                h_val = resg_c - 128
                h0 += np.uint32(h_val == 0); s0 += np.uint64(abs(h_val))

                # 2. RD/BD Channels (Lag)
                if not is_grayscale:
                    target_j = j - 1
                    val1, val2 = rd_ch[i, target_j], bd_ch[i, target_j]
                    a1, b1, c1 = (rd_ch[i, target_j-1] if target_j > 0 else np.uint8(0)), (rd_ch[i-1, target_j] if i > 0 else np.uint8(0)), (rd_ch[i-1, target_j-1] if (i > 0 and target_j > 0) else np.uint8(0))
                    a2, b2, c2 = (bd_ch[i, target_j-1] if target_j > 0 else np.uint8(0)), (bd_ch[i-1, target_j] if i > 0 else np.uint8(0)), (bd_ch[i-1, target_j-1] if (i > 0 and target_j > 0) else np.uint8(0))
                    ctx1 = int(get_context_id_fast(a1, b1, c1, prev_valg, shard_map, nsid))
                    ctx2 = int(get_context_id_fast(a2, b2, c2, prev_valg, shard_map, nsid))
                    p1 = predict_med_standard(a1, b1, c1)
                    p2 = predict_med_standard(a2, b2, c2)
                    diff1 = (int(val1) - int(p1)) & 0xFF
                    diff2 = (int(val2) - int(p2)) & 0xFF
                    # [v6.5] Centered storage for linear median calculation (0 -> 128)
                    res1_c, res2_c = (diff1 + 128) & 0xFF, (diff2 + 128) & 0xFF
                    local_hists[1, ctx1, res1_c] += 1; row_ptrs[i, 1, ctx1] += 1
                    local_hists[2, ctx2, res2_c] += 1; row_ptrs[i, 2, ctx2] += 1
                    h1_val, h2_val = (diff1 + 128) % 256 - 128, (diff2 + 128) % 256 - 128
                    h1 += np.uint32(h1_val == 0); s1 += np.uint64(abs(h1_val))
                    h2 += np.uint32(h2_val == 0); s2 += np.uint64(abs(h2_val))

                prev_valg = curr_valg

            # [Unrolled End] Final Pixel j=w-1: RD/BD-Lag Only
            if w > 0 and not is_grayscale:
                target_j = w - 1
                val1, val2 = rd_ch[i, target_j], bd_ch[i, target_j]
                a1, b1, c1 = (rd_ch[i, target_j-1] if target_j > 0 else np.uint8(0)), (rd_ch[i-1, target_j] if i > 0 else np.uint8(0)), (rd_ch[i-1, target_j-1] if (i > 0 and target_j > 0) else np.uint8(0))
                a2, b2, c2 = (bd_ch[i, target_j-1] if target_j > 0 else np.uint8(0)), (bd_ch[i-1, target_j] if i > 0 else np.uint8(0)), (bd_ch[i-1, target_j-1] if (i > 0 and target_j > 0) else np.uint8(0))
                ctx1 = int(get_context_id_fast(a1, b1, c1, prev_valg, shard_map, nsid))
                ctx2 = int(get_context_id_fast(a2, b2, c2, prev_valg, shard_map, nsid))
                p1 = predict_med_standard(a1, b1, c1)
                p2 = predict_med_standard(a2, b2, c2)
                diff1 = (int(val1) - int(p1)) & 0xFF
                diff2 = (int(val2) - int(p2)) & 0xFF
                # [v6.5] Centered storage for linear median calculation (0 -> 128)
                res1_c, res2_c = (diff1 + 128) & 0xFF, (diff2 + 128) & 0xFF
                local_hists[1, ctx1, res1_c] += 1; row_ptrs[i, 1, ctx1] += 1
                local_hists[2, ctx2, res2_c] += 1; row_ptrs[i, 2, ctx2] += 1
                h1_val, h2_val = (diff1 + 128) % 256 - 128, (diff2 + 128) % 256 - 128
                h1 += np.uint32(h1_val == 0); s1 += np.uint64(abs(h1_val))
                h2 += np.uint32(h2_val == 0); s2 += np.uint64(abs(h2_val))

            row_hits[i, 0], row_hits[i, 1], row_hits[i, 2] = h0, h1, h2
            row_abs_sums[i, 0], row_abs_sums[i, 1], row_abs_sums[i, 2] = s0, s1, s2
        
        chunk_shard_hists[c_idx] = local_hists

    shard_counts_total = np.zeros((3, n_shards), dtype=np.uint32)
    shard_stats_total = np.zeros((3, n_shards, 256), dtype=np.uint32)
    for c_idx in range(num_chunks):
        shard_stats_total += chunk_shard_hists[c_idx]

    for i in range(h):
        for c in range(3):
            for s in range(n_shards):
                shard_counts_total[c, s] += row_ptrs[i, c, s]

    # Global offsets calculation for Pass 2
    shard_offsets = np.zeros((3, n_shards), dtype=np.uint32)
    for c in range(3):
        curr = 0
        for s in range(n_shards):
            shard_offsets[c, s] = curr
            curr += int(shard_counts_total[c, s])

    row_global_offsets = np.zeros((h, 3, n_shards), dtype=np.uint32)
    for c in range(3):
        for s in range(n_shards):
            curr = int(shard_offsets[c, s])
            for i in range(h):
                row_global_offsets[i, c, s] = uint32(curr)
                curr += int(row_ptrs[i, c, s])

    # [v6.7] Pure BICC Legacy: No Median normalization, only Offset (Min) Tightening.
    shard_medians = np.full((3, n_shards), 128, dtype=np.uint8)
    return shard_counts_total, shard_stats_total, shard_offsets, row_global_offsets, shard_medians, (row_hits.sum(axis=0), row_abs_sums.sum(axis=0))

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def predict_pass_2(h: int, w: int, gr_ch: npt.NDArray[np.uint8], rd_ch: npt.NDArray[np.uint8],
                   bd_ch: npt.NDArray[np.uint8], a_ch: npt.NDArray[np.uint8], is_rgba: bool, is_grayscale: bool,
                   shard_map: npt.NDArray[np.uint8], v_bounds: npt.NDArray[np.uint8],
                   i_segs: npt.NDArray[np.uint8], nsid: int,
                   row_global_offsets: npt.NDArray[np.uint32],
                   shard_medians: npt.NDArray[np.uint8],
                   shard_gr: npt.NDArray[np.uint8], shard_rd: npt.NDArray[np.uint8], shard_bd: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], Tuple[np.uint64, np.float64]]:
    """
    Stage 2: O(N) Encoding Payload Construction.
    
    Architecture (Median-Normalized Sharding):
    Retraces the exact staggered SIMD path as Pass 1, but utilizes the `shard_medians` array
    derived mathematically from the histograms. It subtracts the median distance from the raw 
    residual before applying ZigZag encoding, effectively shifting the peak of the Laplacian
    curve for every shard to exactly 0. 
    
    Fills the contiguous Numba-optimized memory blocks which will be heavily compressed by rANS.
    """
    res_a: npt.NDArray[np.uint8] = np.zeros((h, w), dtype=np.uint8) if is_rgba else np.zeros((1, 1), dtype=np.uint8)
    row_a_hits = np.zeros(h, dtype=np.uint64)
    row_a_sums = np.zeros(h, dtype=np.float64)

    for i in prange(h):
        # Local ptrs initialized from pre-calculated row_global_offsets
        local_ptr_gr = row_global_offsets[i, 0].copy()
        local_ptr_rd = row_global_offsets[i, 1].copy()
        local_ptr_bd = row_global_offsets[i, 2].copy()
        
        curr_valg = np.uint8(0)
        prev_valg = np.uint8(0)
        
        # [Unrolled Start] Pixel j=0: G-Lead Only
        if w > 0:
            ag = np.uint8(0)
            bg = (gr_ch[i-1, 0] if i > 0 else np.uint8(0))
            cg = np.uint8(0)
            pg = predict_med_standard(ag, bg, cg)
            curr_valg = gr_ch[i, 0]
            ctxg = int(get_context_id_fast(ag, bg, cg, pg, shard_map, nsid))
            # [v6.1] Median Normalization ONLY
            diffg = int(curr_valg) - int(pg) - (int(shard_medians[0, ctxg]) - 128)
            shard_gr[local_ptr_gr[ctxg]] = to_zigzag(diffg)
            local_ptr_gr[ctxg] += 1
            prev_valg = curr_valg

        # [Main Loop] j=1 to w-1: Dual-Channel Operation
        for j in range(1, w):
            # 1. G-Channel (Lead)
            ag = gr_ch[i, j-1]
            bg = (gr_ch[i-1, j] if i > 0 else np.uint8(0))
            cg = (gr_ch[i-1, j-1] if i > 0 else np.uint8(0))
            pg = predict_med_standard(ag, bg, cg)
            curr_valg = gr_ch[i, j]
            ctxg = int(get_context_id_fast(ag, bg, cg, pg, shard_map, nsid))
            # [v6.1] Median Normalization ONLY
            diffg = int(curr_valg) - int(pg) - (int(shard_medians[0, ctxg]) - 128)
            shard_gr[local_ptr_gr[ctxg]] = to_zigzag(diffg)
            local_ptr_gr[ctxg] += 1

            # 2. RD/BD Channels (Lag)
            if not is_grayscale:
                target_j = j - 1
                val1, val2 = rd_ch[i, target_j], bd_ch[i, target_j]
                a1, b1, c1 = (rd_ch[i, target_j-1] if target_j > 0 else np.uint8(0)), (rd_ch[i-1, target_j] if i > 0 else np.uint8(0)), (rd_ch[i-1, target_j-1] if (i > 0 and target_j > 0) else np.uint8(0))
                a2, b2, c2 = (bd_ch[i, target_j-1] if target_j > 0 else np.uint8(0)), (bd_ch[i-1, target_j] if i > 0 else np.uint8(0)), (bd_ch[i-1, target_j-1] if (i > 0 and target_j > 0) else np.uint8(0))
                ctx1 = int(get_context_id_fast(a1, b1, c1, prev_valg, shard_map, nsid))
                ctx2 = int(get_context_id_fast(a2, b2, c2, prev_valg, shard_map, nsid))
                p1 = predict_med_standard(a1, b1, c1)
                p2 = predict_med_standard(a2, b2, c2)
                # [v6.1] Median Normalization ONLY
                diff1 = int(val1) - int(p1) - (int(shard_medians[1, ctx1]) - 128)
                diff2 = int(val2) - int(p2) - (int(shard_medians[2, ctx2]) - 128)
                shard_rd[local_ptr_rd[ctx1]] = to_zigzag(diff1)
                shard_bd[local_ptr_bd[ctx2]] = to_zigzag(diff2)
                local_ptr_rd[ctx1] += 1; local_ptr_bd[ctx2] += 1

            prev_valg = curr_valg

        # [Unrolled End] Final Pixel j=w-1: RD/BD-Lag Only
        if w > 0 and not is_grayscale:
            target_j = w - 1
            val1, val2 = rd_ch[i, target_j], bd_ch[i, target_j]
            a1, b1, c1 = (rd_ch[i, target_j-1] if target_j > 0 else np.uint8(0)), (rd_ch[i-1, target_j] if i > 0 else np.uint8(0)), (rd_ch[i-1, target_j-1] if (i > 0 and target_j > 0) else np.uint8(0))
            a2, b2, c2 = (bd_ch[i, target_j-1] if target_j > 0 else np.uint8(0)), (bd_ch[i-1, target_j] if i > 0 else np.uint8(0)), (bd_ch[i-1, target_j-1] if (i > 0 and target_j > 0) else np.uint8(0))
            ctx1 = int(get_context_id_fast(a1, b1, c1, prev_valg, shard_map, nsid))
            ctx2 = int(get_context_id_fast(a2, b2, c2, prev_valg, shard_map, nsid))
            p1 = predict_med_standard(a1, b1, c1)
            p2 = predict_med_standard(a2, b2, c2)
            # [v6.1] Median Normalization ONLY
            diff1 = int(val1) - int(p1) - (int(shard_medians[1, ctx1]) - 128)
            diff2 = int(val2) - int(p2) - (int(shard_medians[2, ctx2]) - 128)
            shard_rd[local_ptr_rd[ctx1]] = to_zigzag(diff1)
            shard_bd[local_ptr_bd[ctx2]] = to_zigzag(diff2)
            local_ptr_rd[ctx1] += 1; local_ptr_bd[ctx2] += 1

        if is_rgba:
            # Simple Alpha sharding (non-dynamic for throughput)
            ag_a, cg_a = uint8(0), uint8(0)
            row_src = a_ch[i]
            row_tsrc = a_ch[i-1] if i > 0 else row_src # placeholder if i=0
            h_acc, s_acc = uint64(0), 0.0
            for j in range(w):
                bg_a = row_tsrc[j] if i > 0 else uint8(0)
                val_a = row_src[j]
                pg_a = predict_med_standard(ag_a, bg_a, cg_a)
                r_a = to_zigzag(int(val_a) - int(pg_a))
                res_a[i, j] = r_a; s_acc += abs(float(val_a) - float(pg_a)); h_acc += uint64(r_a == 0); ag_a, cg_a = val_a, (row_tsrc[j] if i > 0 else uint8(0))
            row_a_hits[i], row_a_sums[i] = h_acc, s_acc

    return res_a, (uint64(row_a_hits.sum()), row_a_sums.sum())

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def reconstruct_channels(h, w, res_gr, res_rd, res_bd, off_gr, off_rd, off_bd,
                         shard_counts, shard_medians, is_grayscale,
                         shard_map, v_bounds, i_segs, nsid):
    """
    Stage 3 (Decompression): Parallel Channel Reconstruction Engine. (Phase 2b)
    """
    gr_rec = np.empty((h, w), dtype=np.uint8)
    rd_rec = np.zeros((h, w), dtype=np.uint8) if not is_grayscale else np.zeros((1, 1), dtype=np.uint8)
    bd_rec = np.zeros((h, w), dtype=np.uint8) if not is_grayscale else np.zeros((1, 1), dtype=np.uint8)
    
    ptr_gr, ptr_rd, ptr_bd = off_gr.copy(), off_rd.copy(), off_bd.copy()
    
    for i in range(h):
        for j in range(w):
            ag = gr_rec[i, j - 1] if j > 0 else np.uint8(0)
            bg = gr_rec[i-1, j] if i > 0 else np.uint8(0)
            cg = gr_rec[i-1, j-1] if (i > 0 and j > 0) else np.uint8(0)
            pg = predict_med_standard(ag, bg, cg)
            ctxg = int(get_context_id_fast(ag, bg, cg, pg, shard_map, nsid))
            resg = from_zigzag(res_gr[ptr_gr[ctxg]]); ptr_gr[ctxg] += 1
            gr_rec[i, j] = np.uint8((int(resg) + int(pg) + (int(shard_medians[0, ctxg]) - 128)) & 0xFF)

    # Pass 2: Chroma Dispatch (Parallel Lag)
    if not is_grayscale:
        for c_idx in prange(2):
            if c_idx == 0:
                # Reconstruct RD Channel
                for i in range(h):
                    for j in range(w):
                        cur_g = gr_rec[i, j]
                        a = rd_rec[i, j - 1] if j > 0 else np.uint8(0)
                        b = rd_rec[i-1, j] if i > 0 else np.uint8(0)
                        c = rd_rec[i-1, j-1] if (i > 0 and j > 0) else np.uint8(0)
                        p = predict_med_standard(a, b, c)
                        ctx = int(get_context_id_fast(a, b, c, cur_g, shard_map, nsid))
                        res = from_zigzag(res_rd[ptr_rd[ctx]]); ptr_rd[ctx] += 1
                        rd_rec[i, j] = np.uint8((int(res) + int(p) + (int(shard_medians[1, ctx]) - 128)) & 0xFF)
            else:
                # Reconstruct BD Channel
                for i in range(h):
                    for j in range(w):
                        cur_g = gr_rec[i, j]
                        a = bd_rec[i, j - 1] if j > 0 else np.uint8(0)
                        b = bd_rec[i-1, j] if i > 0 else np.uint8(0)
                        c = bd_rec[i-1, j-1] if (i > 0 and j > 0) else np.uint8(0)
                        p = predict_med_standard(a, b, c)
                        ctx = int(get_context_id_fast(a, b, c, cur_g, shard_map, nsid))
                        res = from_zigzag(res_bd[ptr_bd[ctx]]); ptr_bd[ctx] += 1
                        bd_rec[i, j] = np.uint8((int(res) + int(p) + (int(shard_medians[2, ctx]) - 128)) & 0xFF)

    return gr_rec, rd_rec, bd_rec
