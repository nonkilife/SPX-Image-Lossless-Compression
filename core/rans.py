"""
SPX v8.3.2 [Stable Parallel Architecture]
Module: rans
Role: Pillar 4 - Entropy Engine (Unified).
Description: Integrated 4-way interleaved rANS core with sharded parallel encoding.
Architecture: Precision-indexed SIMD-optimized symbols for O(1) branchless decoding.

Engineering Rationale (4-Way Interleaving):
1. ILP Optimization: Standard rANS is inherently sequential (State = F(State, Symbol)). 
   By interleaving 4 independent streams (4 states), we allow the CPU's out-of-order 
   execution engine to pipeline multiple state updates simultaneously.
2. Branchless Decoding: The decoder core uses a 4096-entry lookup table to eliminate 
   the search for symbols, reducing the inner loop to a series of arithmetic operations 
   that map perfectly to modern super-scalar pipelines.
3. Cache Efficiency: The 4-way structure fits within the L1 instruction cache while 
   processing 4 symbols per iteration, significantly reducing the "Stall-to-Compute" ratio.
4. LIFO Property: rANS is a stack-based (LIFO) engine. Symbols are encoded in 
   forward order and decoded in reverse order (bottom-up), allowing states to be 
   revolved back to their initial base through fractional probability divisions.

Technical Flowchart:
```mermaid
graph TD
    Symbols[Raw Symbols] --> Hist[Histogram Analysis]
    Hist --> PDF[PDF Builder: Precision Normalization]
    PDF --> Encode[4-Way Interleaved Encoder]
    Encode --> Flush[State Flushing -> Bitstream]
    
    Bitstream[Compressed Bitstream] --> Header[Header: Expand PDF Tables]
    Header --> Decode[4-Way Interleaved Decoder]
    Decode --> Reconst[Symbol Reconstruction]
```
"""

__version__ = "8.3.2"

import numpy as np
import numpy.typing as npt
from numba import njit, prange, uint8, uint16, uint32, uint64, int32
from typing import Tuple, List, Optional

from .common import (
    get_empirical_templates,
)
from .rans_selector import _decide_shard_mode_core


# =============================================================================
# --- Global Industrial Constants ---
# =============================================================================
# rANS Precision and Bounds
L_LOWER: uint64 = uint64(2147483648) # 1 << 31 (L_ANS Lower Bound)
M_BITS: int     = 12                  # Probability precision (Total mass = 4096)
L_MAX_BOUND: uint64 = (L_LOWER >> M_BITS) << 8 # Normalization threshold
M_TOTAL: uint64 = uint64(1 << M_BITS)
M_MASK: uint64  = M_TOTAL - 1
SENTINEL_BYTES: int = 8       # Memory safety offset for 64-bit state flushing


# =============================================================================
# --- Core Math Utilities ---
# =============================================================================
@njit(inline='always', cache=True)
def mul_hi(a, b):
    """ Optimized manual 64x64 -> high-64 bit multiplication. """
    ua = np.uint64(a)
    ub = np.uint64(b)
    a_lo = ua & np.uint64(0xFFFFFFFF)
    a_hi = ua >> 32
    b_lo = ub & np.uint64(0xFFFFFFFF)
    b_hi = ub >> 32
    
    p00 = a_lo * b_lo
    p01 = a_lo * b_hi
    p10 = a_hi * b_lo
    p11 = a_hi * b_hi
    
    mid_lo = (p01 & np.uint64(0xFFFFFFFF)) + (p10 & np.uint64(0xFFFFFFFF)) + (p00 >> 32)
    mid_hi = (p01 >> 32) + (p10 >> 32) + (mid_lo >> 32)
    return p11 + mid_hi

# --- 2. 4-Way Interleaved rANS Core ---


@njit(boundscheck=False, cache=True)
def rans_decode_4way_core(st0: uint64, st1: uint64, st2: uint64, st3: uint64, 
                          bitstream: npt.NDArray[np.uint8], 
                          cum_freqs: npt.NDArray[np.uint64], 
                          symbol_freqs: npt.NDArray[np.uint64], 
                          slot_lookup: npt.NDArray[np.uint8], 
                          out: npt.NDArray[np.uint8]) -> None:
    target_len: int = len(out)
    ptr: int = len(bitstream) - 1
    
    num_blocks: int = target_len // 4
    remainder: int = target_len % 4
    
    if symbol_freqs[0] == uint64(4096):
        out.fill(0)
        return
    
    idx: int = 0
    for _ in range(num_blocks):
        # Stream 0
        s0: uint8 = slot_lookup[st0 & M_MASK]
        out[idx] = s0; idx += 1
        st0 = symbol_freqs[s0] * (st0 >> M_BITS) + (st0 & M_MASK) - cum_freqs[s0]
        if st0 < L_LOWER and ptr >= 0:
            st0 = (st0 << 8) | uint64(bitstream[ptr]); ptr -= 1
            if st0 < L_LOWER and ptr >= 0: st0 = (st0 << 8) | uint64(bitstream[ptr]); ptr -= 1

        # Stream 1
        s1: uint8 = slot_lookup[st1 & M_MASK]
        out[idx] = s1; idx += 1
        st1 = symbol_freqs[s1] * (st1 >> M_BITS) + (st1 & M_MASK) - cum_freqs[s1]
        if st1 < L_LOWER and ptr >= 0:
            st1 = (st1 << 8) | uint64(bitstream[ptr]); ptr -= 1
            if st1 < L_LOWER and ptr >= 0: st1 = (st1 << 8) | uint64(bitstream[ptr]); ptr -= 1

        # Stream 2
        s2: uint8 = slot_lookup[st2 & M_MASK]
        out[idx] = s2; idx += 1
        st2 = symbol_freqs[s2] * (st2 >> M_BITS) + (st2 & M_MASK) - cum_freqs[s2]
        if st2 < L_LOWER and ptr >= 0:
            st2 = (st2 << 8) | uint64(bitstream[ptr]); ptr -= 1
            if st2 < L_LOWER and ptr >= 0: st2 = (st2 << 8) | uint64(bitstream[ptr]); ptr -= 1

        # Stream 3
        s3: uint8 = slot_lookup[st3 & M_MASK]
        out[idx] = s3; idx += 1
        st3 = symbol_freqs[s3] * (st3 >> M_BITS) + (st3 & M_MASK) - cum_freqs[s3]
        if st3 < L_LOWER and ptr >= 0:
            st3 = (st3 << 8) | uint64(bitstream[ptr]); ptr -= 1
            if st3 < L_LOWER and ptr >= 0: st3 = (st3 << 8) | uint64(bitstream[ptr]); ptr -= 1

    if remainder > 0:
        if remainder >= 1:
            s_rem0: uint8 = slot_lookup[st0 & M_MASK]; out[idx] = s_rem0; idx += 1
            st0 = symbol_freqs[s_rem0] * (st0 >> M_BITS) + (st0 & M_MASK) - cum_freqs[s_rem0]
            if st0 < L_LOWER and ptr >= 0:
                st0 = (st0 << 8) | uint64(bitstream[ptr]); ptr -= 1
                if st0 < L_LOWER and ptr >= 0: st0 = (st0 << 8) | uint64(bitstream[ptr]); ptr -= 1
        if remainder >= 2:
            s_rem1: uint8 = slot_lookup[st1 & M_MASK]; out[idx] = s_rem1; idx += 1
            st1 = symbol_freqs[s_rem1] * (st1 >> M_BITS) + (st1 & M_MASK) - cum_freqs[s_rem1]
            if st1 < L_LOWER and ptr >= 0:
                st1 = (st1 << 8) | uint64(bitstream[ptr]); ptr -= 1
                if st1 < L_LOWER and ptr >= 0: st1 = (st1 << 8) | uint64(bitstream[ptr]); ptr -= 1
        if remainder >= 3:
            s_rem2: uint8 = slot_lookup[st2 & M_MASK]; out[idx] = s_rem2; idx += 1
            st2 = symbol_freqs[s_rem2] * (st2 >> M_BITS) + (st2 & M_MASK) - cum_freqs[s_rem2]
            if st2 < L_LOWER and ptr >= 0:
                st2 = (st2 << 8) | uint64(bitstream[ptr]); ptr -= 1
                if st2 < L_LOWER and ptr >= 0: st2 = (st2 << 8) | uint64(bitstream[ptr]); ptr -= 1


@njit(fastmath=True, cache=True)
def collect_freqs_jit(data: npt.NDArray[np.uint8], freqs_out: npt.NDArray[np.uint64]):
    for i in range(len(data)):
        freqs_out[data[i]] += uint64(1)

@njit(fastmath=True, cache=True)
def build_pdf_tables_from_shards_core(shard_hists: npt.NDArray[np.uint64],
                                    shard_widths: npt.NDArray[np.uint16],
                                    templates: npt.NDArray[np.uint64]) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64], npt.NDArray[np.uint8]]:
    """ Builds cumulative frequency tables for all shards in a single JIT pass. """
    num_shards: int = shard_hists.shape[0]
    all_sym_freqs: npt.NDArray[np.uint64] = np.zeros((num_shards, 256), dtype=np.uint64)
    all_cum_freqs: npt.NDArray[np.uint64] = np.zeros((num_shards, 257), dtype=np.uint64)
    shard_modes: npt.NDArray[np.uint8] = np.zeros(num_shards, dtype=np.uint8)
    
    for sid in range(num_shards):
        width = int(shard_widths[sid])
        h_vals = shard_hists[sid]
        
        if np.sum(h_vals) > 0:
            best_mode, f_arr = _decide_shard_mode_core(h_vals, width, 120.0, templates, False)
            shard_modes[sid] = best_mode
            all_sym_freqs[sid] = f_arr
            
            acc = uint64(0)
            for j in range(256):
                all_cum_freqs[sid, j] = acc
                acc += f_arr[j]
            all_cum_freqs[sid, 256] = acc
        else:
            shard_modes[sid] = uint8(3)
            all_sym_freqs[sid, 0] = uint64(4096)
            all_cum_freqs[sid, 1:] = uint64(4096)
            
    return all_cum_freqs, all_sym_freqs, shard_modes

def build_pdf_tables_from_shards(shard_buffers: List[npt.NDArray[np.uint8]], 
                                 shard_widths: npt.NDArray[np.uint16]) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64], npt.NDArray[np.uint8]]:
    num_shards: int = len(shard_buffers)
    shard_hists = np.zeros((num_shards, 256), dtype=np.uint64)
    for sid in range(num_shards):
        if len(shard_buffers[sid]) > 0:
            collect_freqs_jit(shard_buffers[sid], shard_hists[sid])
    
    templates = get_empirical_templates()
    return build_pdf_tables_from_shards_core(shard_hists, shard_widths, templates)

def compact_pdf_tables(all_sym_freqs: npt.NDArray[np.uint64], shard_widths: npt.NDArray[np.uint16], shard_modes: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """ 
    Serializes custom frequency tables into a compact bytes-buffer for the .spx Header.

    Routing Logic:
    - Modes 4-9 (Static Empirical Templates) and Mode 3 (Uniform/Empty) are skipped entirely.
      They consume 0 bytes in the payload block since they are hardcoded in the Decoder.
    - Mode 0 (Custom Dynamic): Takes the generated PDF and decides on the most physically 
      compact byte-representation before writing:
        [Dense Internal Mode 0]: Writes every probability sequentially up to `shard_width` (Cost: w*2 bytes).
        [Sparse Internal Mode 1]: Only writes probabilities > 0 alongside their offsets (Cost: 2+n*3 bytes).
    """
    num_shards = all_sym_freqs.shape[0]
    payloads = []
    
    for s in range(num_shards):
        mode = shard_modes[s]
        if mode >= 3: # Mode 3 (Empty) or 4-9 (Templates)
            continue
            
        w = int(shard_widths[s])
        indices = np.where(all_sym_freqs[s] > 0)[0].astype(np.uint8)
        n = len(indices)
        
        # Mode internal to PDF block
        cost_dense = w * 2
        cost_sparse = 2 + (n * 3)
        
        if cost_dense <= cost_sparse:
            payloads.append(uint8(0).tobytes() + all_sym_freqs[s, :w].astype(np.uint16).tobytes())
        else:
            payloads.append(uint8(1).tobytes() + np.uint16(n).tobytes() + indices.tobytes() + all_sym_freqs[s, indices].astype(np.uint16).tobytes())
            
    return np.frombuffer(b"".join(payloads), dtype=np.uint8)

def expand_pdf_tables(compacted_data: npt.NDArray[np.uint8], shard_widths: npt.NDArray[np.uint16], shard_modes: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint64]:
    """ 
    Reconstructs the 256-symbol probability arrays during decompression.

    Routing Logic:
    - Intercepts Modes 4-9 and retrieves the Static Template from local memory without consuming IO bytes.
    - Intercepts Mode 3 and synthesizes a [4096, 0, 0...] perfect hit array.
    - For Custom PDFs (Mode 0), it decodes the payload stream (Dense vs Sparse byte-shapes) up to
      the limits defined by `shard_widths`.
    """
    num_shards = len(shard_widths)
    expanded = np.zeros((num_shards, 256), dtype=np.uint64)
    data_bytes = compacted_data.tobytes()
    ptr = 0
    
    templates = get_empirical_templates()
    
    for s in range(num_shards):
        mode = shard_modes[s]
        
        if mode >= 4:
            # Empirical template: retrieve hardcoded PDF, no bitstream bytes consumed
            tid = mode - 4
            expanded[s] = templates[tid].astype(np.uint64)
        elif mode == 3:
            expanded[s, 0] = uint64(4096)
        else:
            if ptr >= len(data_bytes): 
                expanded[s, 0] = uint64(4096); continue
                
            internal_mode = data_bytes[ptr]; ptr += 1
            if internal_mode == 0:
                w = int(shard_widths[s])
                needed = w * 2
                if ptr + needed > len(data_bytes): 
                    # Attempt recovery
                    expanded[s, 0] = uint64(4096); continue
                freqs = np.frombuffer(data_bytes[ptr:ptr + needed], dtype=np.uint16)
                ptr += needed
                expanded[s, :w] = freqs.astype(np.uint64)
            elif internal_mode == 1:
                n_v = np.frombuffer(data_bytes[ptr:ptr+2], dtype=np.uint16)
                if len(n_v) == 0: expanded[s, 0] = uint64(4096); continue
                n = int(n_v[0]); ptr += 2
                needed = n + (n * 2)
                if ptr + needed > len(data_bytes): 
                    expanded[s, 0] = uint64(4096); continue
                symbols = np.frombuffer(data_bytes[ptr:ptr+n], dtype=np.uint8); ptr += n
                freqs = np.frombuffer(data_bytes[ptr:ptr+n*2], dtype=np.uint16); ptr += n*2
                expanded[s, symbols] = freqs.astype(np.uint64)

    return expanded

@njit(parallel=True, fastmath=True, cache=True)
def rans_encode_shards_parallel(shard_data_flat: npt.NDArray[np.uint8], 
                                shard_offsets: npt.NDArray[np.uint32], 
                                shard_lengths: npt.NDArray[np.uint32], 
                                all_cum_freqs: npt.NDArray[np.uint64], 
                                all_sym_freqs: npt.NDArray[np.uint64], 
                                initial_state: uint64) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint8], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    num_shards: int = len(shard_lengths)
    final_states: npt.NDArray[np.uint64] = np.zeros((num_shards, 4), dtype=np.uint64)
    bs_lengths: npt.NDArray[np.uint32] = np.zeros(num_shards, dtype=np.uint32)
    
    total_data_len: int = shard_data_flat.shape[0]
    bitstreams_flat: npt.NDArray[np.uint8] = np.empty(total_data_len * 2 + (1024 * num_shards), dtype=np.uint8)
    bs_offsets: npt.NDArray[np.uint32] = np.zeros(num_shards, dtype=np.uint32)
    
    curr_bs_offset: uint32 = uint32(0)
    for i in range(num_shards):
        bs_offsets[i] = curr_bs_offset
        curr_bs_offset += uint32(shard_lengths[i] * 2 + 1024)
    
    all_magics = np.empty((num_shards, 256), dtype=uint64)
    for i in prange(num_shards):
        sf = all_sym_freqs[i]
        for s_idx in range(256):
            f_v = sf[s_idx]
            if f_v > 0: 
                all_magics[i, s_idx] = uint64(0xFFFFFFFFFFFFFFFF) // f_v
            else: 
                all_magics[i, s_idx] = uint64(0)

    for i in prange(num_shards):
        n_val = int(shard_lengths[i])
        sfreqs = all_sym_freqs[i]
        
        if n_val == 0 or sfreqs[0] == uint64(4096):
            final_states[i, 0] = initial_state
            final_states[i, 1] = initial_state
            final_states[i, 2] = initial_state
            final_states[i, 3] = initial_state
        else:
            data_start = int(shard_offsets[i])
            bs_start = int(bs_offsets[i])
            
            st0 = initial_state
            st1 = initial_state
            st2 = initial_state
            st3 = initial_state
            ptr = bs_start + SENTINEL_BYTES
            
            bitstreams_flat[bs_start : ptr].fill(0)
            
            cfreqs = all_cum_freqs[i]
            magics = all_magics[i]
            
            rem_val = n_val % 4
            tail_start = n_val - rem_val
            
            for j in range(n_val - 1, tail_start - 1, -1):
                s_val = int(shard_data_flat[data_start + j])
                f_val = sfreqs[s_val]
                cf_val = cfreqs[s_val]
                m_val = magics[s_val]
                
                pos_val = j % 4
                curr_st = uint64(0)
                if pos_val == 0: curr_st = st0
                elif pos_val == 1: curr_st = st1
                elif pos_val == 2: curr_st = st2
                else: curr_st = st3
                
                x_max_val = L_MAX_BOUND * f_val
                while curr_st >= x_max_val:
                    bitstreams_flat[ptr] = uint8(curr_st & 0xFF); ptr += 1
                    curr_st >>= 8
                
                q = mul_hi(curr_st, m_val)
                r = curr_st - q * f_val
                if r >= f_val:
                    q += np.uint64(1)
                    r -= f_val
                curr_st = (q << M_BITS) + cf_val + r
                
                if pos_val == 0: st0 = curr_st
                elif pos_val == 1: st1 = curr_st
                elif pos_val == 2: st2 = curr_st
                else: st3 = curr_st

            for j in range(tail_start - 4, -1, -4):
                s3_val = int(shard_data_flat[data_start + j + 3])
                f3_val = sfreqs[s3_val]
                cf3_val = cfreqs[s3_val]
                m3_val = magics[s3_val]
                while st3 >= L_MAX_BOUND * f3_val:
                    bitstreams_flat[ptr] = uint8(st3 & 0xFF); ptr += 1; st3 >>= 8
                q3 = mul_hi(st3, m3_val); r3 = st3 - q3 * f3_val
                if r3 >= f3_val: q3 += np.uint64(1); r3 -= f3_val
                st3 = (q3 << M_BITS) + cf3_val + r3

                s2_val = int(shard_data_flat[data_start + j + 2])
                f2_val = sfreqs[s2_val]
                cf2_val = cfreqs[s2_val]
                m2_val = magics[s2_val]
                while st2 >= L_MAX_BOUND * f2_val:
                    bitstreams_flat[ptr] = uint8(st2 & 0xFF); ptr += 1; st2 >>= 8
                q2 = mul_hi(st2, m2_val); r2 = st2 - q2 * f2_val
                if r2 >= f2_val: q2 += np.uint64(1); r2 -= f2_val
                st2 = (q2 << M_BITS) + cf2_val + r2

                s1_val = int(shard_data_flat[data_start + j + 1])
                f1_val = sfreqs[s1_val]
                cf1_val = cfreqs[s1_val]
                m1_val = magics[s1_val]
                while st1 >= L_MAX_BOUND * f1_val:
                    bitstreams_flat[ptr] = uint8(st1 & 0xFF); ptr += 1; st1 >>= 8
                q1 = mul_hi(st1, m1_val); r1 = st1 - q1 * f1_val
                if r1 >= f1_val: q1 += np.uint64(1); r1 -= f1_val
                st1 = (q1 << M_BITS) + cf1_val + r1

                s0_val = int(shard_data_flat[data_start + j])
                f0_val = sfreqs[s0_val]
                cf0_val = cfreqs[s0_val]
                m0_val = magics[s0_val]
                while st0 >= L_MAX_BOUND * f0_val:
                    bitstreams_flat[ptr] = uint8(st0 & 0xFF); ptr += 1; st0 >>= 8
                q0 = mul_hi(st0, m0_val); r0 = st0 - q0 * f0_val
                if r0 >= f0_val: q0 += np.uint64(1); r0 -= f0_val
                st0 = (q0 << M_BITS) + cf0_val + r0

            final_states[i, 0] = st0
            final_states[i, 1] = st1
            final_states[i, 2] = st2
            final_states[i, 3] = st3
            bs_lengths[i] = uint32(ptr - bs_start)
        
    return final_states, bitstreams_flat, bs_offsets, bs_lengths

@njit(parallel=True, cache=True)
def build_all_lookups(all_cum_freqs: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint8]:
    """ Batch-precomputes O(1) slot-lookup tables for all shards in parallel. """
    n_ch = all_cum_freqs.shape[0]
    n_sh = all_cum_freqs.shape[1]
    lookups = np.zeros((n_ch, n_sh, 4096), dtype=uint8)
    
    for c in range(n_ch):
        for s in prange(n_sh):
            cf = all_cum_freqs[c, s]
            lk = lookups[c, s]
            for sym in range(256):
                start = cf[sym]
                end = cf[sym+1]
                if end > start:
                    lk[start:end] = uint8(sym)
    return lookups

@njit(parallel=True, boundscheck=False, cache=True)
def rans_decode_shards_parallel(
    compressed_data: npt.NDArray[np.uint8],
    states: npt.NDArray[np.uint64],        
    bs_offsets: npt.NDArray[np.uint32],    
    bs_lengths: npt.NDArray[np.uint32],    
    all_cum_freqs: npt.NDArray[np.uint64], 
    all_sym_freqs: npt.NDArray[np.uint64], 
    all_lookups: npt.NDArray[np.uint8],    
    out_flat: npt.NDArray[np.uint8],
    out_offsets: npt.NDArray[np.uint32],   
    shard_counts: npt.NDArray[np.uint32]   
) -> None:
    """ Flattened Parallel Decoder Dispatcher: Saturates all CPU cores across shards. """
    num_shards = len(shard_counts)
    for i in prange(num_shards):
        target_cnt = int(shard_counts[i])
        if target_cnt > 0:
            st0, st1, st2, st3 = states[i, 0], states[i, 1], states[i, 2], states[i, 3]
            bs_start = int(bs_offsets[i])
            bs_len = int(bs_lengths[i])
            out_start = int(out_offsets[i])
            
            b_stream = compressed_data[bs_start : bs_start + bs_len]
            out_view = out_flat[out_start : out_start + target_cnt]
            
            rans_decode_4way_core(
                st0, st1, st2, st3, 
                b_stream, 
                all_cum_freqs[i], 
                all_sym_freqs[i], 
                all_lookups[i], 
                out_view
            )

