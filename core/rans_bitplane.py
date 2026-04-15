"""
ZPNG-CSDE v6.6 [Bitplane rANS Engine]
Module: rans_bitplane
Role: High-precision grayscale entropy coding.
Description: 2-bit interleaved contextual rANS engine utilizing 4-layer parallel sharding.

Technical Flowchart:
```mermaid
graph TD
    In[8-bit Grayscale Residuals] --> Split[Bitplane Splitting: 4 x 2-bit Layers]
    Split --> Shard[Vertical Sharding: 4 Strips]
    
    subgraph "Per Shard Pipeline"
        S1[Context Extraction: L/U/NW Neighbors] --> S2[Frequency Estimation: 64 Contexts/Layer]
        S2 --> S3[rANS Interleaved Encoder: 4-way Parallel]
    end
    
    S3 --> Pack[Bitstream Packing with Metadata]
    Pack --> Out[ZPNG Bitplane Payload]
```
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange, uint8, uint16, uint32, uint64
from typing import Tuple, List, Optional

# =============================================================================
# --- Configuration ---
# =============================================================================
BITPLANE_N_SHARDS: int = 4 

# =============================================================================
# --- 2-Bit Contextual rANS Engine (Bitplane Shards) ---
# =============================================================================
@njit(cache=True, boundscheck=False, fastmath=True, nogil=True)
def rans_decode_2bit_4way_kernel(bitstream: npt.NDArray[np.uint8], 
                                st0: uint64, st1: uint64, st2: uint64, st3: uint64, 
                                h: int, w: int, 
                                all_cf: npt.NDArray[np.uint64], 
                                all_sf: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint8]:
    """ Fixed interleaved decoder with zero-allocation inner loop. """
    rec: npt.NDArray[np.uint8] = np.zeros((h, w), dtype=uint8)
    s0, s1, s2, s3 = uint64(st0), uint64(st1), uint64(st2), uint64(st3)
    
    l_lower, m_bits, mask = uint64(1 << 31), uint64(12), uint64((1 << 12) - 1)
    ptr: int = len(bitstream) - 1
    
    for y in range(h):
        # L (Left neighbors) initialized at start of row
        l0, l1, l2, l3 = uint8(0), uint8(0), uint8(0), uint8(0)
        prev_up: uint8 = uint8(0)  # cached nw = up from previous x-iteration
        for x in range(w):
            if y == 0:
                ctx0, ctx1, ctx2, ctx3 = l0, l1, l2, l3
            else:
                up: uint8 = rec[y-1, x]
                nw: uint8 = prev_up  # was rec[y-1, x-1]; 0 at x==0 by initialization
                u0, u1, u2, u3 = up&0x03, (up>>2)&0x03, (up>>4)&0x03, (up>>6)&0x03
                n0, n1, n2, n3 = nw&0x03, (nw>>2)&0x03, (nw>>4)&0x03, (nw>>6)&0x03
                ctx0 = l0 | (u0 << 2) | (n0 << 4)
                ctx1 = l1 | (u1 << 2) | (n1 << 4)
                ctx2 = l2 | (u2 << 2) | (n2 << 4)
                ctx3 = l3 | (u3 << 2) | (n3 << 4)
                prev_up = up
            
            # Layer 0
            cf0, sf0 = all_cf[0, ctx0], all_sf[0, ctx0]
            slot0 = s0 & mask; sym0 = uint8((slot0 >= cf0[1]) + (slot0 >= cf0[2]) + (slot0 >= cf0[3]))
            s0 = sf0[sym0] * (s0 >> m_bits) + (slot0 - cf0[sym0])
            if s0 < l_lower and ptr >= 0:
                s0 = (s0 << 8) | uint64(bitstream[ptr]); ptr -= 1
                if s0 < l_lower and ptr >= 0: s0 = (s0 << 8) | uint64(bitstream[ptr]); ptr -= 1

            # Layer 1
            cf1, sf1 = all_cf[1, ctx1], all_sf[1, ctx1]
            slot1 = s1 & mask; sym1 = uint8((slot1 >= cf1[1]) + (slot1 >= cf1[2]) + (slot1 >= cf1[3]))
            s1 = sf1[sym1] * (s1 >> m_bits) + (slot1 - cf1[sym1])
            if s1 < l_lower and ptr >= 0:
                s1 = (s1 << 8) | uint64(bitstream[ptr]); ptr -= 1
                if s1 < l_lower and ptr >= 0: s1 = (s1 << 8) | uint64(bitstream[ptr]); ptr -= 1

            # Layer 2
            cf2, sf2 = all_cf[2, ctx2], all_sf[2, ctx2]
            slot2 = s2 & mask; sym2 = uint8((slot2 >= cf2[1]) + (slot2 >= cf2[2]) + (slot2 >= cf2[3]))
            s2 = sf2[sym2] * (s2 >> m_bits) + (slot2 - cf2[sym2])
            if s2 < l_lower and ptr >= 0:
                s2 = (s2 << 8) | uint64(bitstream[ptr]); ptr -= 1
                if s2 < l_lower and ptr >= 0: s2 = (s2 << 8) | uint64(bitstream[ptr]); ptr -= 1

            # Layer 3
            cf3, sf3 = all_cf[3, ctx3], all_sf[3, ctx3]
            slot3 = s3 & mask; sym3 = uint8((slot3 >= cf3[1]) + (slot3 >= cf3[2]) + (slot3 >= cf3[3]))
            s3 = sf3[sym3] * (s3 >> m_bits) + (slot3 - cf3[sym3])
            if s3 < l_lower and ptr >= 0:
                s3 = (s3 << 8) | uint64(bitstream[ptr]); ptr -= 1
                if s3 < l_lower and ptr >= 0: s3 = (s3 << 8) | uint64(bitstream[ptr]); ptr -= 1

            rec[y, x] = sym0 | (sym1 << 2) | (sym2 << 4) | (sym3 << 6)
            l0, l1, l2, l3 = sym0, sym1, sym2, sym3
            
    return rec

@njit(parallel=True, cache=True, boundscheck=False, fastmath=True, nogil=True)
def decompress_shards_parallel(payload: npt.NDArray[np.uint8], h: int, w: int, 
                               n_shards: int, shard_h: int, 
                               pdfs_f: npt.NDArray[np.uint64], 
                               pdfs_cf: npt.NDArray[np.uint64], 
                               states: npt.NDArray[np.uint64], 
                               lens: npt.NDArray[np.uint32], 
                               off: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint8]:
    """ Orchestrates parallel decompression across vertical image shards. """
    # Max feasible height for any shard (handle remainder correctly)
    max_h = shard_h + (h % n_shards) + 1
    shard_results: npt.NDArray[np.uint8] = np.zeros((n_shards, max_h, w), dtype=uint8) 
    for s_idx in prange(n_shards):
        sh: int = shard_h if s_idx < n_shards-1 else h - (s_idx*shard_h)
        bs: npt.NDArray[np.uint8] = payload[off[s_idx] : off[s_idx] + lens[s_idx]]
        res: npt.NDArray[np.uint8] = rans_decode_2bit_4way_kernel(
            bs, states[s_idx,0], states[s_idx,1], states[s_idx,2], states[s_idx,3], sh, w, pdfs_cf, pdfs_f
        )
        shard_results[s_idx, :sh, :] = res
    return shard_results

def decompress_bitplane_gray(payload: bytes, h: int, w: int) -> npt.NDArray[np.uint8]:
    """ Main entry point for restoring bitplane-encoded grayscale streams. """
    ptr: int = 0
    raw_payload: npt.NDArray[np.uint8] = np.frombuffer(payload, dtype=np.uint8)
    
    n_shards: int = int(raw_payload[ptr]); ptr += 1
    pdfs_f_list: List[npt.NDArray[np.uint64]] = []
    pdfs_cf_list: List[npt.NDArray[np.uint64]] = []
    
    # Read global frequency tables for each of the 4 layers
    for _ in range(4):
        f: npt.NDArray[np.uint64] = np.frombuffer(payload[ptr:ptr+512], dtype=np.uint16).reshape((64, 4)).astype(np.uint64)
        ptr += 512
        cf: npt.NDArray[np.uint64] = np.zeros((64, 5), dtype=np.uint64)
        for i in range(64): 
            cf[i, 1:] = np.cumsum(f[i])
        pdfs_f_list.append(f)
        pdfs_cf_list.append(cf)
        
    # Read serialized rANS end-states (Total: n_shards * 4 layers * 8 bytes)
    states: npt.NDArray[np.uint64] = np.frombuffer(payload[ptr : ptr + n_shards * 32], dtype=np.uint64).reshape((n_shards, 4))
    ptr += n_shards * 32
    
    # Read bitstream lengths per shard
    lens: npt.NDArray[np.uint32] = np.frombuffer(payload[ptr : ptr + n_shards * 4], dtype=np.uint32)
    ptr += n_shards * 4
    
    # Map bitstream offsets
    off: npt.NDArray[np.uint32] = np.zeros(n_shards, dtype=np.uint32)
    curr: uint32 = uint32(ptr)
    for i in range(n_shards): 
        off[i] = curr; curr += uint32(lens[i])
    
    shard_h: int = h // n_shards
    shard_results: npt.NDArray[np.uint8] = decompress_shards_parallel(
        raw_payload, h, w, n_shards, shard_h, 
        np.stack(pdfs_f_list), np.stack(pdfs_cf_list), states, lens, off
    )
    
    # Assemble shards into final image
    res: npt.NDArray[np.uint8] = np.zeros((h, w), dtype=np.uint8)
    for s_idx in range(n_shards):
        y0: int = s_idx * shard_h
        y1: int = (s_idx+1) * shard_h if s_idx < n_shards-1 else h
        res[y0:y1, :] = shard_results[s_idx, :y1-y0, :]
    return res

@njit(cache=True, boundscheck=False, nogil=True)
def get_2bit_contexts_opt(plane: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """ Optimized extraction of 6-bit spatial contexts (L|U<<2|NW<<4). """
    h, w = plane.shape
    contexts: npt.NDArray[np.uint8] = np.zeros(h * w, dtype=uint8)
    for y in range(h):
        row_curr = plane[y]
        row_ctx = contexts[y*w : (y+1)*w]
        if y == 0:
            for x in range(w): 
                row_ctx[x] = row_curr[x-1] if x > 0 else uint8(0)
        else:
            row_prev = plane[y-1]
            for x in range(w):
                l_val: uint8 = row_curr[x-1] if x > 0 else uint8(0)
                u_val: uint8 = row_prev[x]
                nw_val: uint8 = row_prev[x-1] if x > 0 else uint8(0)
                row_ctx[x] = l_val | (u_val << 2) | (nw_val << 4)
    return contexts

@njit(cache=True)
def build_pdf_2bit_global(symbols_all: npt.NDArray[np.uint8], contexts_all: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """ Quantizes context histograms into 12-bit rANS frequency tables. """
    counts: npt.NDArray[np.uint64] = np.zeros((64, 4), dtype=uint64)
    for i in range(len(symbols_all)): 
        counts[contexts_all[i], symbols_all[i]] += 1
    
    f_all: npt.NDArray[np.uint64] = np.zeros((64, 4), dtype=uint64)
    cf_all: npt.NDArray[np.uint64] = np.zeros((64, 5), dtype=uint64)
    precision: uint64 = uint64(4096)
    
    for c in range(64):
        ctx_counts = counts[c]
        t = np.sum(ctx_counts)
        if t > 0:
            nf = np.round(ctx_counts * precision / t).astype(np.uint64)
            for i in range(4):
                if nf[i] == 0:
                    nf[i] = uint64(1)
            diff = int(precision) - int(np.sum(nf))
            peak_i = 0
            for i in range(1, 4):
                if nf[i] > nf[peak_i]:
                    peak_i = i
            nf[peak_i] = uint64(int(nf[peak_i]) + diff)
            f_all[c], cf_all[c, 1:] = nf, np.cumsum(nf)
        else: 
            # Fallback for unused contexts: Uniform distribution
            f_all[c] = uint64(1024)
            cf_all[c, 1:] = np.array([1024, 2048, 3072, 4096], dtype=np.uint64)
    return f_all, cf_all

@njit(inline='always', cache=True)
def _mul_hi_bp(a: uint64, b: uint64) -> uint64:
    """ 64x64 -> high-64 multiply for magic-number division. """
    a_lo = a & np.uint64(0xFFFFFFFF); a_hi = a >> np.uint64(32)
    b_lo = b & np.uint64(0xFFFFFFFF); b_hi = b >> np.uint64(32)
    p00 = a_lo * b_lo; p01 = a_lo * b_hi; p10 = a_hi * b_lo; p11 = a_hi * b_hi
    mid_lo = (p01 & np.uint64(0xFFFFFFFF)) + (p10 & np.uint64(0xFFFFFFFF)) + (p00 >> np.uint64(32))
    return p11 + (p01 >> np.uint64(32)) + (p10 >> np.uint64(32)) + (mid_lo >> np.uint64(32))

@njit(cache=True, boundscheck=False, nogil=True)
def rans_encode_2bit_4way_kernel(shard_data: npt.NDArray[np.uint8],
                                all_cf: npt.NDArray[np.uint64],
                                all_sf: npt.NDArray[np.uint64]) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint8]]:
    """
    Cross-Layer 4-Way Interleaved rANS Encoder.
    Processes pixel data in reverse scan order to satisfy decoder front-pull requirements.
    """
    h, w = shard_data.shape
    st0, st1, st2, st3 = uint64(1 << 31), uint64(1 << 31), uint64(1 << 31), uint64(1 << 31)
    l_lower, m_bits = uint64(1 << 31), uint64(12)
    l_max_bound: uint64 = (l_lower >> m_bits) << 8

    # Precompute magic constants: magic[layer, ctx, sym] = (2^64-1) // freq
    # Replaces per-symbol integer division with a multiply + 1-step correction.
    magic: npt.NDArray[np.uint64] = np.zeros((4, 64, 4), dtype=np.uint64)
    for _l in range(4):
        for _c in range(64):
            for _s in range(4):
                _f = all_sf[_l, _c, _s]
                if _f > uint64(0):
                    magic[_l, _c, _s] = uint64(0xFFFFFFFFFFFFFFFF) // _f

    out: npt.NDArray[np.uint8] = np.zeros(h * w * 4, dtype=uint8)
    ptr: int = 0

    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            px: uint8 = shard_data[y, x]
            s0, s1, s2, s3 = px&0x03, (px>>2)&0x03, (px>>4)&0x03, (px>>6)&0x03

            # Derive 4 contexts for the current pixel layers
            if y == 0:
                lpx: uint8 = shard_data[y, x-1] if x > 0 else uint8(0)
                c0, c1, c2, c3 = lpx&0x03, (lpx>>2)&0x03, (lpx>>4)&0x03, (lpx>>6)&0x03
            else:
                up: uint8 = shard_data[y-1, x]
                nw: uint8 = shard_data[y-1, x-1] if x > 0 else uint8(0)
                lpx: uint8 = shard_data[y, x-1] if x > 0 else uint8(0)

                # Formula: L | (U << 2) | (NW << 4)
                c0 = (lpx&0x03) | ((up&0x03) << 2) | ((nw&0x03) << 4)
                c1 = ((lpx>>2)&0x03) | (((up>>2)&0x03) << 2) | (((nw>>2)&0x03) << 4)
                c2 = ((lpx>>4)&0x03) | (((up>>4)&0x03) << 2) | (((nw>>4)&0x03) << 4)
                c3 = ((lpx>>6)&0x03) | (((up>>6)&0x03) << 2) | (((nw>>6)&0x03) << 4)

            # Sequence-critical: Encode all 4 layers in reverse (3 then 2 then 1 then 0)

            # Layer 3
            f3 = all_sf[3, c3, s3]; cf3 = all_cf[3, c3, s3]
            while st3 >= l_max_bound * f3:
                out[ptr] = uint8(st3 & 0xFF); st3 >>= uint64(8); ptr += 1
            q3 = _mul_hi_bp(st3, magic[3, c3, s3]); r3 = st3 - q3 * f3
            if r3 >= f3: q3 += uint64(1); r3 -= f3
            st3 = (q3 << m_bits) + r3 + cf3

            # Layer 2
            f2 = all_sf[2, c2, s2]; cf2 = all_cf[2, c2, s2]
            while st2 >= l_max_bound * f2:
                out[ptr] = uint8(st2 & 0xFF); st2 >>= uint64(8); ptr += 1
            q2 = _mul_hi_bp(st2, magic[2, c2, s2]); r2 = st2 - q2 * f2
            if r2 >= f2: q2 += uint64(1); r2 -= f2
            st2 = (q2 << m_bits) + r2 + cf2

            # Layer 1
            f1 = all_sf[1, c1, s1]; cf1 = all_cf[1, c1, s1]
            while st1 >= l_max_bound * f1:
                out[ptr] = uint8(st1 & 0xFF); st1 >>= uint64(8); ptr += 1
            q1 = _mul_hi_bp(st1, magic[1, c1, s1]); r1 = st1 - q1 * f1
            if r1 >= f1: q1 += uint64(1); r1 -= f1
            st1 = (q1 << m_bits) + r1 + cf1

            # Layer 0
            f0 = all_sf[0, c0, s0]; cf0 = all_cf[0, c0, s0]
            while st0 >= l_max_bound * f0:
                out[ptr] = uint8(st0 & 0xFF); st0 >>= uint64(8); ptr += 1
            q0 = _mul_hi_bp(st0, magic[0, c0, s0]); r0 = st0 - q0 * f0
            if r0 >= f0: q0 += uint64(1); r0 -= f0
            st0 = (q0 << m_bits) + r0 + cf0
                
    final_res: npt.NDArray[np.uint64] = np.zeros(4, dtype=uint64)
    final_res[0], final_res[1], final_res[2], final_res[3] = st0, st1, st2, st3
    return final_res, out[:ptr]

def compress_bitplane_gray(raw_data: npt.NDArray[np.uint8]) -> bytes:
    """ Primary orchestrator for bitplane grayscale compression. """
    h, w = raw_data.shape
    shard_h: int = h // BITPLANE_N_SHARDS
    
    # Pre-calculate layers and global PDFs
    layers: List[npt.NDArray[np.uint8]] = [(raw_data >> (i*2)) & 0x03 for i in range(4)]
    pdfs_f_list: List[npt.NDArray[np.uint64]] = []
    pdfs_cf_list: List[npt.NDArray[np.uint64]] = []
    
    for l_idx in range(4):
        ctx: npt.NDArray[np.uint8] = get_2bit_contexts_opt(layers[l_idx])
        f, cf = build_pdf_2bit_global(layers[l_idx].flatten(), ctx)
        pdfs_f_list.append(f)
        pdfs_cf_list.append(cf)
    
    # Use stack array for Numba parallel calls
    stack_cf: npt.NDArray[np.uint64] = np.stack(pdfs_cf_list)
    stack_f: npt.NDArray[np.uint64] = np.stack(pdfs_f_list)
    
    results: List[Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint8]]] = []
    for s_idx in range(BITPLANE_N_SHARDS):
        y0: int = s_idx * shard_h
        y1: int = (s_idx+1) * shard_h if s_idx < BITPLANE_N_SHARDS-1 else h
        shard_data = raw_data[y0:y1, :]
        results.append(rans_encode_2bit_4way_kernel(shard_data, stack_cf, stack_f))
    
    # Construct final payload
    out = bytearray()
    out.append(BITPLANE_N_SHARDS)
    # 1. Global Frequency Tables (for each layer)
    for f in pdfs_f_list: 
        out.extend(f.astype(np.uint16).tobytes())
    # 2. Final rANS States
    for r in results: 
        out.extend(r[0].tobytes()) 
    # 3. Bitstream Lengths
    for r in results: 
        out.extend(np.array([len(r[1])], dtype=np.uint32).tobytes())
    # 4. Concatenated Bitstreams
    for r in results: 
        out.extend(r[1].tobytes())
        
    return bytes(out)
