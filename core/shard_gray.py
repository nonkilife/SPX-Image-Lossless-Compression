"""
SPX [Grayscale Sharding Module]
Module: shard_gray
Role: Pillar 3.5 (Grayscale) - Single-Channel Data Partitioning.
Description: N-shard BICC orchestration for grayscale images.

Design Philosophy: High-Throughput Sequential Raster
----------------------------------------------------
Unlike the RGB path, the Grayscale path ignores cross-channel coordination and 
utilizes a raw, sequential raster scan. This eliminates the 'Staggered Step' 
logic and the associated branch overhead, allowing the Numba-JIT compiler to 
maximize loop-level parallelism and SIMD throughput.

Process Flow:
1. Unified Raster Scan: Pixels are processed in standard [i, j] order.
2. Context Selection: Each pixel derives its shard ID using spatial neighbors (a, b, c) 
   and the local predictor (p) for intensity classification.
3. Decoupled Architecture: While currently using the Universal-42 profile for 
   system stability, this module is profile-agnostic and ready to support 
   specialized monochrome profiles (e.g., Gray-Fast-30).

Logic Path Visualization:
```mermaid
graph TD
    A[Raw Gray Channel] --> B[predict_pass_1: Sequential Profiling]
    B --> C{Decision Hub}
    C -->|Standard| D[predict_pass_2: Shard-Flat Residuals]
    C -->|Bitplane| E[Standard Bitplane Path]
    D --> F[Flat Shard Buffers - rANS]
```

Engineering Note: Output Compatibility
--------------------------------------
To maintain compatibility with the orchestrator (compress.py), the grayscale path 
returns 3-channel data structures where channel 0 is populated and channels 
1-2 are left null.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange, uint8, uint64
from typing import Tuple
from .common import to_zigzag, selected_predictor
from .sharding import get_context_id_fast


@njit(parallel=True, fastmath=True, error_model='numpy')
def predict_pass_1_gray(h: int, w: int, gray_ch: npt.NDArray[np.uint8],
                        shard_map: npt.NDArray[np.uint8], nsid: int,
                        s_lut: npt.NDArray[np.uint8], i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint8], Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint64]]]:
    """
    Stage 1: O(N) Shard Profiling & Histogram Generation for grayscale.

    Simplified from shard_rgb.predict_pass_1: single channel, standard raster
    scan (no BICC stagger). Output arrays use the same 3-channel shapes as
    predict_pass_1 for drop-in compatibility with compress.py - channel 0 is
    populated, channels 1 and 2 are left as zeros.
    """
    n_shards = int(shard_map.max()) + 1 if nsid < 0 else nsid + 1
    num_chunks = int(min(16, h)) if h > 0 else 1
    chunk_size = (h + num_chunks - 1) // num_chunks
    chunk_shard_hists = np.zeros((num_chunks, n_shards, 256), dtype=np.uint32)

    row_ptrs = np.zeros((h, n_shards), dtype=np.uint32)
    row_hits = np.zeros(h, dtype=np.uint32)
    row_abs_sums = np.zeros(h, dtype=np.uint64)

    for c_idx in prange(num_chunks):
        start_i = c_idx * chunk_size
        end_i = min(start_i + chunk_size, h)
        local_hists = np.zeros((n_shards, 256), dtype=np.uint32)

        for i in range(start_i, end_i):
            pi = i + 1  # padded row index (gray_ch is (h+2, w+2))
            h_acc = np.uint32(0)
            s_acc = np.uint64(0)

            for pj in range(1, w + 1):
                a = gray_ch[pi, pj-1]
                b = gray_ch[pi-1, pj]
                c = gray_ch[pi-1, pj-1]
                p = selected_predictor(a, b, c)
                val = gray_ch[pi, pj]
                # Luma Context: Uses PREDICTOR as intensity baseline (self-referential)
                ctx = int(get_context_id_fast(a, b, c, p, s_lut, i_lut, d_lut))
                diff = (int(val) - int(p)) & 0xFF
                # Centered storage: 0 residual maps to 128
                res_c = (diff + 128) & 0xFF
                local_hists[ctx, res_c] += 1
                row_ptrs[i, ctx] += 1
                h_val = res_c - 128
                h_acc += np.uint32(h_val == 0)
                s_acc += np.uint64(abs(h_val))

            row_hits[i] = h_acc
            row_abs_sums[i] = s_acc

        chunk_shard_hists[c_idx] = local_hists

    # Reduce chunk histograms
    shard_stats_1ch = np.zeros((n_shards, 256), dtype=np.uint32)
    for c_idx in range(num_chunks):
        shard_stats_1ch += chunk_shard_hists[c_idx]

    shard_counts_1ch = np.zeros(n_shards, dtype=np.uint32)
    for i in range(h):
        for s in range(n_shards):
            shard_counts_1ch[s] += row_ptrs[i, s]

    # Global offsets for Pass 2
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

    # --- Wrap into 3-channel shapes for downstream compatibility ---
    shard_counts_out = np.zeros((3, n_shards), dtype=np.uint32)
    shard_counts_out[0] = shard_counts_1ch

    shard_stats_out = np.zeros((3, n_shards, 256), dtype=np.uint32)
    shard_stats_out[0] = shard_stats_1ch

    shard_offsets_out = np.zeros((3, n_shards), dtype=np.uint32)
    shard_offsets_out[0] = shard_offsets_1ch

    row_global_offsets_out = np.zeros((h, 3, n_shards), dtype=np.uint32)
    for i in range(h):
        row_global_offsets_out[i, 0] = row_global_offsets_1ch[i]

    hits_out = np.zeros(3, dtype=np.uint32)
    hits_out[0] = row_hits.sum()
    sums_out = np.zeros(3, dtype=np.uint64)
    sums_out[0] = row_abs_sums.sum()

    return shard_counts_out, shard_stats_out, shard_offsets_out, row_global_offsets_out, (hits_out, sums_out)


@njit(parallel=True, fastmath=True, error_model='numpy')
def predict_pass_2_gray(h: int, w: int, gray_ch: npt.NDArray[np.uint8],
                        a_ch: npt.NDArray[np.uint8], is_rgba: bool,
                        shard_map: npt.NDArray[np.uint8], nsid: int,
                        s_lut: npt.NDArray[np.uint8], i_lut: npt.NDArray[np.uint8], d_lut: npt.NDArray[np.uint8],
                        row_global_offsets: npt.NDArray[np.uint32],
                        shard_out: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], Tuple[np.uint64, np.float64]]:
    """
    Stage 2: O(N) Encoding Payload Construction for grayscale.

    Retraces the same raster scan as Pass 1, and writes ZigZag-encoded 
    residuals into the pre-allocated shard_out buffer using the 
    row_global_offsets write-pointer table (channel 0 slice).

    Alpha channel handling mirrors shard_rgb.predict_pass_2 for RGBA images.
    """
    res_a: npt.NDArray[np.uint8] = np.zeros((h, w), dtype=np.uint8) if is_rgba else np.zeros((1, 1), dtype=np.uint8)
    row_a_hits = np.zeros(h, dtype=np.uint64)
    row_a_sums = np.zeros(h, dtype=np.float64)

    for i in prange(h):
        pi = i + 1  # padded row index (gray_ch is (h+2, w+2))
        local_ptr = row_global_offsets[i, 0].copy()

        for pj in range(1, w + 1):
            a = gray_ch[pi, pj-1]
            b = gray_ch[pi-1, pj]
            c = gray_ch[pi-1, pj-1]
            p = selected_predictor(a, b, c)
            val = gray_ch[pi, pj]
            ctx = int(get_context_id_fast(a, b, c, p, s_lut, i_lut, d_lut))
            # Residual calculation
            diff = int(val) - int(p)
            shard_out[local_ptr[ctx]] = to_zigzag(diff)
            local_ptr[ctx] += 1

        if is_rgba:
            ag_a, cg_a = uint8(0), uint8(0)
            row_src = a_ch[i]
            row_tsrc = a_ch[i-1] if i > 0 else row_src
            h_acc, s_acc = uint64(0), 0.0
            for j in range(w):
                bg_a = row_tsrc[j] if i > 0 else uint8(0)
                val_a = row_src[j]
                pg_a = selected_predictor(ag_a, bg_a, cg_a)
                r_a = to_zigzag(int(val_a) - int(pg_a))
                res_a[i, j] = r_a
                s_acc += abs(float(val_a) - float(pg_a))
                h_acc += uint64(r_a == 0)
                ag_a, cg_a = val_a, (row_tsrc[j] if i > 0 else uint8(0))
            row_a_hits[i], row_a_sums[i] = h_acc, s_acc

    return res_a, (uint64(row_a_hits.sum()), row_a_sums.sum())
