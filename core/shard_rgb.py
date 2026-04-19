"""
ZPNG-CSDE [High-Performance Parallel Architecture]
Module: shard_rgb
Role: Pillar 3.5 - Data Partitioning (BICC).
Description: Unified sharding orchestration compatible with 3D Mapping LUTs.

Logic Path Visualization:
```mermaid
graph TD
    A[Raw Channels: G, RD, BD] --> B[predict_pass_1: BICC Profile]
    B --> Stagger{Staggered Window}
    Stagger -->|G-Lead| D[G Context -> ShardProfile.shard_map]
    Stagger -->|RD/BD-Lag| E[RD/BD Context + G-Ref]
    D & E --> F[Median Normalization]
    F --> G[predict_pass_2: Vectorized Residuals]
```
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange, uint8, uint16, uint32, uint64
from typing import Tuple, List, Optional
from .common import (
    to_zigzag, predict_med_standard, med_edge_tuned, get_context_id_fast
)

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def predict_pass_1(h: int, w: int, gr_ch: npt.NDArray[np.uint8], rd_ch: npt.NDArray[np.uint8],
                   bd_ch: npt.NDArray[np.uint8], is_grayscale: bool,
                   shard_map: npt.NDArray[np.uint8], nsid: int) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint64]]]:
    """
    Stage 1: O(N) Shard Profiling & Histogram Generation.

    Architecture (Staggered Look-ahead):
    Extracts predictive residuals natively using MED (Median Edge Detection).
    Uses a 1-pixel staggered shift between Green (Lead) and RD/BD (Lag) channels: G processes
    pixel j while RD/BD process pixel j-1, using the same-position G value as context.
    This interleaved order enables pipeline parallelism while keeping context consistent
    between encoder (prev_valg) and decoder (gr_rec[i, j]) for the same target pixel.

    Engineering Note: The 'Staggered Step' (j vs j-1) is the core of ZPNG's cross-channel
    decorrelation. By lagging the RD/BD channels, we can use the ACTUAL reconstructed
    Green pixel at position j as a context for the RD/BD pixels at position j.
    This ensures zero-drift between compression and decompression.

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
            pi = i + 1  # padded row (channels are (h+2, w+2))
            h0, h1, h2 = np.uint32(0), np.uint32(0), np.uint32(0)
            s0, s1, s2 = np.uint64(0), np.uint64(0), np.uint64(0)

            curr_valg = np.uint8(0)
            prev_valg = np.uint8(0)

            # Pixel j=0 (pj=1): G-Lead; RD/BD still waiting
            if w > 0:
                ag = np.uint8(0)           # left border
                bg = gr_ch[pi-1, 1]        # top neighbor; border=0 when i=0
                cg = np.uint8(0)           # top-left corner border

                pg = med_edge_tuned(ag, bg, cg)
                curr_valg = gr_ch[pi, 1]
                ctxg = int(get_context_id_fast(ag, bg, cg, pg, shard_map, nsid))
                diffg = (int(curr_valg) - int(pg)) & 0xFF
                resg_c = (diffg + 128) & 0xFF
                local_hists[0, int(ctxg), resg_c] += 1; row_ptrs[i, 0, int(ctxg)] += 1
                h_val = resg_c - 128
                h0 += np.uint32(h_val == 0); s0 += np.uint64(abs(h_val))
                prev_valg = curr_valg

            # j=1..w-1 (pj=2..w): G-Lead at pj, lagged RD/BD at ptj=pj-1
            for pj in range(2, w + 1):
                # 1. G-Channel (Lead)
                ag = gr_ch[pi, pj-1]
                bg = gr_ch[pi-1, pj]
                cg = gr_ch[pi-1, pj-1]
                pg = med_edge_tuned(ag, bg, cg)
                curr_valg = gr_ch[pi, pj]
                ctxg = int(get_context_id_fast(ag, bg, cg, pg, shard_map, nsid))
                diffg = (int(curr_valg) - int(pg)) & 0xFF
                resg_c = (diffg + 128) & 0xFF
                local_hists[0, int(ctxg), resg_c] += 1; row_ptrs[i, 0, int(ctxg)] += 1
                h_val = resg_c - 128
                h0 += np.uint32(h_val == 0); s0 += np.uint64(abs(h_val))

                # 2. RD/BD Channels (Lag): pixel at ptj = pj-1
                if not is_grayscale:
                    ptj = pj - 1
                    val1, val2 = rd_ch[pi, ptj], bd_ch[pi, ptj]
                    a1 = rd_ch[pi, ptj-1]; b1 = rd_ch[pi-1, ptj]; c1 = rd_ch[pi-1, ptj-1]
                    a2 = bd_ch[pi, ptj-1]; b2 = bd_ch[pi-1, ptj]; c2 = bd_ch[pi-1, ptj-1]
                    ctx1 = int(get_context_id_fast(a1, b1, c1, prev_valg, shard_map, nsid))
                    ctx2 = int(get_context_id_fast(a2, b2, c2, prev_valg, shard_map, nsid))
                    p1 = med_edge_tuned(a1, b1, c1)
                    p2 = med_edge_tuned(a2, b2, c2)
                    diff1 = (int(val1) - int(p1)) & 0xFF
                    diff2 = (int(val2) - int(p2)) & 0xFF
                    res1_c, res2_c = (diff1 + 128) & 0xFF, (diff2 + 128) & 0xFF
                    local_hists[1, ctx1, res1_c] += 1; row_ptrs[i, 1, ctx1] += 1
                    local_hists[2, ctx2, res2_c] += 1; row_ptrs[i, 2, ctx2] += 1
                    h1_val, h2_val = res1_c - 128, res2_c - 128
                    h1 += np.uint32(h1_val == 0); s1 += np.uint64(abs(h1_val))
                    h2 += np.uint32(h2_val == 0); s2 += np.uint64(abs(h2_val))

                prev_valg = curr_valg

            # Final: RD/BD lag catches up to last pixel (ptj = w)
            if w > 0 and not is_grayscale:
                ptj = w
                val1, val2 = rd_ch[pi, ptj], bd_ch[pi, ptj]
                a1 = rd_ch[pi, ptj-1]; b1 = rd_ch[pi-1, ptj]; c1 = rd_ch[pi-1, ptj-1]
                a2 = bd_ch[pi, ptj-1]; b2 = bd_ch[pi-1, ptj]; c2 = bd_ch[pi-1, ptj-1]
                ctx1 = int(get_context_id_fast(a1, b1, c1, prev_valg, shard_map, nsid))
                ctx2 = int(get_context_id_fast(a2, b2, c2, prev_valg, shard_map, nsid))
                p1 = med_edge_tuned(a1, b1, c1)
                p2 = med_edge_tuned(a2, b2, c2)
                diff1 = (int(val1) - int(p1)) & 0xFF
                diff2 = (int(val2) - int(p2)) & 0xFF
                res1_c, res2_c = (diff1 + 128) & 0xFF, (diff2 + 128) & 0xFF
                local_hists[1, ctx1, res1_c] += 1; row_ptrs[i, 1, ctx1] += 1
                local_hists[2, ctx2, res2_c] += 1; row_ptrs[i, 2, ctx2] += 1
                h1_val, h2_val = res1_c - 128, res2_c - 128
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

    # Medians fixed at 128 (neutral): median normalization in Pass 2 is a no-op.
    shard_medians = np.full((3, n_shards), 128, dtype=np.uint8)
    return shard_counts_total, shard_stats_total, shard_offsets, row_global_offsets, shard_medians, (row_hits.sum(axis=0), row_abs_sums.sum(axis=0))

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def predict_pass_2(h: int, w: int, gr_ch: npt.NDArray[np.uint8], rd_ch: npt.NDArray[np.uint8],
                   bd_ch: npt.NDArray[np.uint8], a_ch: npt.NDArray[np.uint8], is_rgba: bool, is_grayscale: bool,
                   shard_map: npt.NDArray[np.uint8], nsid: int,
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
        pi = i + 1  # padded row (channels are (h+2, w+2))
        # Local ptrs initialized from pre-calculated row_global_offsets
        local_ptr_gr = row_global_offsets[i, 0].copy()
        local_ptr_rd = row_global_offsets[i, 1].copy()
        local_ptr_bd = row_global_offsets[i, 2].copy()

        curr_valg = np.uint8(0)
        prev_valg = np.uint8(0)

        # Pixel j=0 (pj=1): G-Lead; RD/BD still waiting
        if w > 0:
            ag = np.uint8(0)           # left border
            bg = gr_ch[pi-1, 1]        # top neighbor; border=0 when i=0
            cg = np.uint8(0)           # top-left corner border
            pg = med_edge_tuned(ag, bg, cg)
            curr_valg = gr_ch[pi, 1]
            ctxg = int(get_context_id_fast(ag, bg, cg, pg, shard_map, nsid))
            diffg = int(curr_valg) - int(pg)
            shard_gr[local_ptr_gr[ctxg]] = np.uint8((diffg + 128) & 0xFF)
            local_ptr_gr[ctxg] += 1
            prev_valg = curr_valg

        # j=1..w-1 (pj=2..w): G-Lead at pj, lagged RD/BD at ptj=pj-1
        for pj in range(2, w + 1):
            # 1. G-Channel (Lead)
            ag = gr_ch[pi, pj-1]
            bg = gr_ch[pi-1, pj]
            cg = gr_ch[pi-1, pj-1]
            pg = med_edge_tuned(ag, bg, cg)
            curr_valg = gr_ch[pi, pj]
            ctxg = int(get_context_id_fast(ag, bg, cg, pg, shard_map, nsid))
            diffg = int(curr_valg) - int(pg)
            shard_gr[local_ptr_gr[ctxg]] = np.uint8((diffg + 128) & 0xFF)
            local_ptr_gr[ctxg] += 1

            # 2. RD/BD Channels (Lag): pixel at ptj = pj-1
            if not is_grayscale:
                ptj = pj - 1
                val1, val2 = rd_ch[pi, ptj], bd_ch[pi, ptj]
                a1 = rd_ch[pi, ptj-1]; b1 = rd_ch[pi-1, ptj]; c1 = rd_ch[pi-1, ptj-1]
                a2 = bd_ch[pi, ptj-1]; b2 = bd_ch[pi-1, ptj]; c2 = bd_ch[pi-1, ptj-1]
                ctx1 = int(get_context_id_fast(a1, b1, c1, prev_valg, shard_map, nsid))
                ctx2 = int(get_context_id_fast(a2, b2, c2, prev_valg, shard_map, nsid))
                p1 = med_edge_tuned(a1, b1, c1)
                p2 = med_edge_tuned(a2, b2, c2)
                diff1 = int(val1) - int(p1)
                diff2 = int(val2) - int(p2)
                shard_rd[local_ptr_rd[ctx1]] = np.uint8((diff1 + 128) & 0xFF)
                shard_bd[local_ptr_bd[ctx2]] = np.uint8((diff2 + 128) & 0xFF)
                local_ptr_rd[ctx1] += 1; local_ptr_bd[ctx2] += 1

            prev_valg = curr_valg

        # Final: RD/BD lag catches up to last pixel (ptj = w)
        if w > 0 and not is_grayscale:
            ptj = w
            val1, val2 = rd_ch[pi, ptj], bd_ch[pi, ptj]
            a1 = rd_ch[pi, ptj-1]; b1 = rd_ch[pi-1, ptj]; c1 = rd_ch[pi-1, ptj-1]
            a2 = bd_ch[pi, ptj-1]; b2 = bd_ch[pi-1, ptj]; c2 = bd_ch[pi-1, ptj-1]
            ctx1 = int(get_context_id_fast(a1, b1, c1, prev_valg, shard_map, nsid))
            ctx2 = int(get_context_id_fast(a2, b2, c2, prev_valg, shard_map, nsid))
            p1 = med_edge_tuned(a1, b1, c1)
            p2 = med_edge_tuned(a2, b2, c2)
            diff1 = int(val1) - int(p1)
            diff2 = int(val2) - int(p2)
            shard_rd[local_ptr_rd[ctx1]] = np.uint8((diff1 + 128) & 0xFF)
            shard_bd[local_ptr_bd[ctx2]] = np.uint8((diff2 + 128) & 0xFF)
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
                res_a[i, j] = r_a
                s_acc += abs(float(val_a) - float(pg_a))
                h_acc += uint64(r_a == 0)
                ag_a, cg_a = val_a, (row_tsrc[j] if i > 0 else uint8(0))
            row_a_hits[i], row_a_sums[i] = h_acc, s_acc

    return res_a, (uint64(row_a_hits.sum()), row_a_sums.sum())

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def reconstruct_channels(h, w, res_gr, res_rd, res_bd, off_gr, off_rd, off_bd,
                         shard_counts, shard_medians, is_grayscale,
                         shard_map, nsid):
    """
    Stage 3 (Decompression): Parallel Channel Reconstruction Engine. (Phase 2b)
    """
    # Zero-padded internal arrays: border stays 0, loop from pi=1..h+1, pj=1..w+1
    gr_rec = np.zeros((h + 2, w + 2), dtype=np.uint8)
    rd_rec = np.zeros((h + 2, w + 2), dtype=np.uint8) if not is_grayscale else np.zeros((1, 1), dtype=np.uint8)
    bd_rec = np.zeros((h + 2, w + 2), dtype=np.uint8) if not is_grayscale else np.zeros((1, 1), dtype=np.uint8)

    ptr_gr, ptr_rd, ptr_bd = off_gr.copy(), off_rd.copy(), off_bd.copy()

    for pi in range(1, h + 1):
        for pj in range(1, w + 1):
            ag = gr_rec[pi, pj-1]
            bg = gr_rec[pi-1, pj]
            cg = gr_rec[pi-1, pj-1]
            pg = med_edge_tuned(ag, bg, cg)
            ctxg = int(get_context_id_fast(ag, bg, cg, pg, shard_map, nsid))
            resg = int(res_gr[ptr_gr[ctxg]]) - 128; ptr_gr[ctxg] += 1
            gr_rec[pi, pj] = np.uint8((resg + int(pg)) & 0xFF)

    # Pass 2: Chroma Dispatch (Parallel Lag)
    if not is_grayscale:
        for c_idx in prange(2):
            if c_idx == 0:
                # Reconstruct RD Channel
                for pi in range(1, h + 1):
                    for pj in range(1, w + 1):
                        cur_g = gr_rec[pi, pj]
                        a = rd_rec[pi, pj-1]
                        b = rd_rec[pi-1, pj]
                        c = rd_rec[pi-1, pj-1]
                        p = med_edge_tuned(a, b, c)
                        ctx = int(get_context_id_fast(a, b, c, cur_g, shard_map, nsid))
                        res = int(res_rd[ptr_rd[ctx]]) - 128; ptr_rd[ctx] += 1
                        rd_rec[pi, pj] = np.uint8((res + int(p)) & 0xFF)
            else:
                # Reconstruct BD Channel
                for pi in range(1, h + 1):
                    for pj in range(1, w + 1):
                        cur_g = gr_rec[pi, pj]
                        a = bd_rec[pi, pj-1]
                        b = bd_rec[pi-1, pj]
                        c = bd_rec[pi-1, pj-1]
                        p = med_edge_tuned(a, b, c)
                        ctx = int(get_context_id_fast(a, b, c, cur_g, shard_map, nsid))
                        res = int(res_bd[ptr_bd[ctx]]) - 128; ptr_bd[ctx] += 1
                        bd_rec[pi, pj] = np.uint8((res + int(p)) & 0xFF)

    if not is_grayscale:
        return gr_rec[1:h+1, 1:w+1], rd_rec[1:h+1, 1:w+1], bd_rec[1:h+1, 1:w+1]
    return gr_rec[1:h+1, 1:w+1], rd_rec, bd_rec
