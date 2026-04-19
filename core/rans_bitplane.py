"""
ZPNG-CSDE [Stable Parallel Architecture]
Module: rans_bitplane
Role: Entropy coding combining N-shard gradient context with
      2-bit spatial bitplane context.

Engineering Rationale:
1. Pillar 4.5 - Residual Decomposition: 8-bit residuals are decomposed into
   four 2-bit layers (Layer 0: bits 0-1, Layer 1: bits 2-3, etc.). This reduces
   the alphabet size from 256 to 4 per operation, simplifying PDF modeling.
2. Temporal-Causal Context: For every pixel, a 6-bit spatial context is derived
   from the already-decoded 2-bit values of the Left (L), Up (U), and North-West (NW)
   neighbors.
3. Shard Conditioning: Allows the engine to adapt to local gradient trends (BICC)
   while maintaining fine-grained spatial awareness.

Technical Architecture:
For every pixel, the combined per-layer context is derived via:
    ctx = shard_id * N_SPATIAL + (L_2bit | U_2bit<<2 | NW_2bit<<4)

Where:
  - shard_id: gradient shard context (N_SHARDS per profile).
  - L/U/NW_2bit: High-order bits of reconstructed spatial neighbors.
  - Context Range: N_SHARDS x N_SPATIAL contexts per layer.

Bitstream Format:
  [0xFF]                          1 byte  - Sharded format magic
  [tables_zstd_len: uint32]       4 bytes - Zstd compressed PDF block length
  [tables_zstd]                   N bytes - Flattened [4, 2688, 4] frequencies
  [states: 4 x uint64]           32 bytes - Final rANS states for 4 layers
  [bs_len: uint32]                4 bytes - Compressed bitstream length
  [bitstream]                    N bytes - Layered rANS payload

Logic Path:
```mermaid
graph TD
    Resid[2D Residuals] --> Layer[Decompose into 4x2-bit Layers]
    Layer --> Ctx[Combine Contexts: 42 Shard x 64 Spatial]
    Ctx --> rANS[Sequential rANS: Layer 3 to 0]
    rANS --> Stream[Bitstream + Zstd PDF Table]
    %% Ctx label: N Shard x 64 Spatial = N_CTX combined contexts
```
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange, uint8, uint64, get_num_threads, get_thread_id
from typing import Tuple
import zstandard as zstd
import concurrent.futures

from .common import predict_med_standard, get_context_id_fast
from .predictor import from_zigzag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SPATIAL: int = 64                         # 2-bit L|U<<2|NW<<4 contexts
N_SHARDS: int  = 42                         # Default for PROFILE_RGB; n_ctx recomputed dynamically per call
N_CTX: int     = N_SHARDS * N_SPATIAL       # Default context count; JIT kernels receive n_ctx at runtime
BITPLANE_MAGIC: int = 0xFF                  # first-byte sentinel

_L_LOWER  = np.uint64(1 << 31)
_M_BITS   = np.uint64(12)
_MASK     = np.uint64((1 << 12) - 1)
_L_MAX    = (_L_LOWER >> _M_BITS) << np.uint64(8)


# ---------------------------------------------------------------------------
# Math helper (inlined by Numba)
# ---------------------------------------------------------------------------
@njit(inline='always', cache=True)
def _mul_hi(a: uint64, b: uint64) -> uint64:
    a_lo = a & np.uint64(0xFFFFFFFF); a_hi = a >> np.uint64(32)
    b_lo = b & np.uint64(0xFFFFFFFF); b_hi = b >> np.uint64(32)
    p00 = a_lo * b_lo; p01 = a_lo * b_hi
    p10 = a_hi * b_lo; p11 = a_hi * b_hi
    mid_lo = ((p01 & np.uint64(0xFFFFFFFF)) + (p10 & np.uint64(0xFFFFFFFF))
              + (p00 >> np.uint64(32)))
    return p11 + (p01 >> np.uint64(32)) + (p10 >> np.uint64(32)) + (mid_lo >> np.uint64(32))


# ---------------------------------------------------------------------------
# Frequency table builder
# ---------------------------------------------------------------------------
@njit(parallel=True, boundscheck=False, cache=True)
def _build_pdf_sharded(resid_2d:  npt.NDArray[np.uint8],
                       gray_ch:   npt.NDArray[np.uint8],
                       shard_map: npt.NDArray[np.uint8],
                       nsid: int,
                       n_ctx: int,
                       nt: int) -> Tuple[npt.NDArray[np.uint64],
                                         npt.NDArray[np.uint64]]:
    """
    Single raster-order pass accumulating symbol counts into
    counts[layer, combined_ctx, symbol], then quantises to 12-bit
    rANS frequencies.

    Returns:
      f  - shape (4, N_CTX, 4) symbol frequencies (sum = 4096 per row)
      cf - shape (4, N_CTX, 5) cumulative frequencies (0-padded on left)
    """
    h, w = resid_2d.shape[0] - 2, resid_2d.shape[1] - 2  # inputs are zero-padded (h+2, w+2)
    counts_tls = np.zeros((nt, 4, n_ctx, 4), dtype=np.uint64)

    for pi in prange(1, h + 1):
        tid = get_thread_id()
        for pj in range(1, w + 1):
            # ----- gradient-shard context from original pixel neighbors -----
            ag = gray_ch[pi, pj-1]
            bg = gray_ch[pi-1, pj]
            cg = gray_ch[pi-1, pj-1]
            p_g = predict_med_standard(ag, bg, cg)
            sid = int(get_context_id_fast(ag, bg, cg, p_g, shard_map, nsid))

            # ----- bitplane spatial context from residual neighbors -----
            r_l = resid_2d[pi, pj-1]
            r_u = resid_2d[pi-1, pj]
            r_n = resid_2d[pi-1, pj-1]
            px  = resid_2d[pi, pj]

            for k in range(4):
                shift = np.uint8(k * 2)
                l_k = (r_l >> shift) & np.uint8(3)
                u_k = (r_u >> shift) & np.uint8(3)
                n_k = (r_n >> shift) & np.uint8(3)
                bp_ctx = int(l_k) | (int(u_k) << 2) | (int(n_k) << 4)
                ctx    = sid * N_SPATIAL + bp_ctx
                sym    = int((px >> shift) & np.uint8(3))
                counts_tls[tid, k, ctx, sym] += np.uint64(1)

    counts = counts_tls.sum(axis=0)

    # ----- quantise to 12-bit rANS tables -----
    precision = np.uint64(4096)
    f  = np.zeros((4, n_ctx, 4), dtype=np.uint64)
    cf = np.zeros((4, n_ctx, 5), dtype=np.uint64)

    for k in range(4):
        for c in range(n_ctx):
            t = np.uint64(0)
            for s in range(4):
                t += counts[k, c, s]

            if t > np.uint64(0):
                nf = np.zeros(4, dtype=np.uint64)
                for s in range(4):
                    nf[s] = np.uint64(int(round(float(counts[k, c, s]) * 4096.0 / float(t))))
                    if nf[s] == np.uint64(0):
                        nf[s] = np.uint64(1)
                acc = np.uint64(0)
                for s in range(4):
                    acc += nf[s]
                diff = int(precision) - int(acc)
                peak = 0
                for s in range(1, 4):
                    if nf[s] > nf[peak]:
                        peak = s
                nf[peak] = np.uint64(int(nf[peak]) + diff)
                acc = np.uint64(0)
                for s in range(4):
                    f[k, c, s]  = nf[s]
                    cf[k, c, s] = acc
                    acc += nf[s]
                cf[k, c, 4] = acc
            else:
                # Uniform fallback for unseen contexts
                for s in range(4):
                    f[k, c, s] = np.uint64(1024)
                cf[k, c, 0] = np.uint64(0)
                cf[k, c, 1] = np.uint64(1024)
                cf[k, c, 2] = np.uint64(2048)
                cf[k, c, 3] = np.uint64(3072)
                cf[k, c, 4] = np.uint64(4096)

    return f, cf


# ---------------------------------------------------------------------------
# Encoder kernel
# ---------------------------------------------------------------------------
@njit(cache=True, boundscheck=False, nogil=True)
def _rans_encode_sharded(resid_2d:  npt.NDArray[np.uint8],
                         gray_ch:   npt.NDArray[np.uint8],
                         all_cf:    npt.NDArray[np.uint64],
                         all_sf:    npt.NDArray[np.uint64],
                         shard_map: npt.NDArray[np.uint8],
                         nsid: int,
                         n_ctx: int) -> Tuple[npt.NDArray[np.uint64],
                                              npt.NDArray[np.uint8]]:
    """
    Reverse-scan 4-way interleaved rANS encoder.
    Layers 3-0 encoded sequentially per pixel (matching decoder pull order).
    Returns (final_states[4], bitstream_bytes).
    """
    h, w = resid_2d.shape[0] - 2, resid_2d.shape[1] - 2  # inputs are zero-padded (h+2, w+2)
    l_lower = np.uint64(1 << 31)
    m_bits  = np.uint64(12)
    l_max   = (l_lower >> m_bits) << np.uint64(8)

    st0 = l_lower; st1 = l_lower; st2 = l_lower; st3 = l_lower

    # Precompute magic constants for branchless division
    magic = np.zeros((4, n_ctx, 4), dtype=np.uint64)
    for k in range(4):
        for c in range(n_ctx):
            for s in range(4):
                fv = all_sf[k, c, s]
                if fv > np.uint64(0):
                    magic[k, c, s] = np.uint64(0xFFFFFFFFFFFFFFFF) // fv

    out = np.empty(h * w * 4 + 64, dtype=np.uint8)
    ptr = 0

    for pi in range(h, 0, -1):
        for pj in range(w, 0, -1):
            # ----- shard ID from original neighbors -----
            ag = gray_ch[pi, pj-1]
            bg = gray_ch[pi-1, pj]
            cg = gray_ch[pi-1, pj-1]
            p_g = predict_med_standard(ag, bg, cg)
            sid = int(get_context_id_fast(ag, bg, cg, p_g, shard_map, nsid))

            # ----- bitplane spatial neighbors -----
            r_l = resid_2d[pi, pj-1]
            r_u = resid_2d[pi-1, pj]
            r_n = resid_2d[pi-1, pj-1]
            px  = resid_2d[pi, pj]

            # ----- Layer 3 -----
            l3 = (r_l >> np.uint8(6)) & np.uint8(3)
            u3 = (r_u >> np.uint8(6)) & np.uint8(3)
            n3 = (r_n >> np.uint8(6)) & np.uint8(3)
            ctx3 = sid * N_SPATIAL + (int(l3) | (int(u3) << 2) | (int(n3) << 4))
            s3   = int((px >> np.uint8(6)) & np.uint8(3))
            f3  = all_sf[3, ctx3, s3];  cf3 = all_cf[3, ctx3, s3]
            m3  = magic[3, ctx3, s3]
            while st3 >= l_max * f3:
                out[ptr] = np.uint8(st3 & np.uint64(0xFF)); ptr += 1
                st3 >>= np.uint64(8)
            q3 = _mul_hi(st3, m3); r3_ = st3 - q3 * f3
            if r3_ >= f3: q3 += np.uint64(1); r3_ -= f3
            st3 = (q3 << m_bits) + r3_ + cf3

            # ----- Layer 2 -----
            l2 = (r_l >> np.uint8(4)) & np.uint8(3)
            u2 = (r_u >> np.uint8(4)) & np.uint8(3)
            n2 = (r_n >> np.uint8(4)) & np.uint8(3)
            ctx2 = sid * N_SPATIAL + (int(l2) | (int(u2) << 2) | (int(n2) << 4))
            s2   = int((px >> np.uint8(4)) & np.uint8(3))
            f2  = all_sf[2, ctx2, s2];  cf2 = all_cf[2, ctx2, s2]
            m2  = magic[2, ctx2, s2]
            while st2 >= l_max * f2:
                out[ptr] = np.uint8(st2 & np.uint64(0xFF)); ptr += 1
                st2 >>= np.uint64(8)
            q2 = _mul_hi(st2, m2); r2_ = st2 - q2 * f2
            if r2_ >= f2: q2 += np.uint64(1); r2_ -= f2
            st2 = (q2 << m_bits) + r2_ + cf2

            # ----- Layer 1 -----
            l1 = (r_l >> np.uint8(2)) & np.uint8(3)
            u1 = (r_u >> np.uint8(2)) & np.uint8(3)
            n1 = (r_n >> np.uint8(2)) & np.uint8(3)
            ctx1 = sid * N_SPATIAL + (int(l1) | (int(u1) << 2) | (int(n1) << 4))
            s1   = int((px >> np.uint8(2)) & np.uint8(3))
            f1  = all_sf[1, ctx1, s1];  cf1 = all_cf[1, ctx1, s1]
            m1  = magic[1, ctx1, s1]
            while st1 >= l_max * f1:
                out[ptr] = np.uint8(st1 & np.uint64(0xFF)); ptr += 1
                st1 >>= np.uint64(8)
            q1 = _mul_hi(st1, m1); r1_ = st1 - q1 * f1
            if r1_ >= f1: q1 += np.uint64(1); r1_ -= f1
            st1 = (q1 << m_bits) + r1_ + cf1

            # ----- Layer 0 -----
            l0 = r_l & np.uint8(3)
            u0 = r_u & np.uint8(3)
            n0 = r_n & np.uint8(3)
            ctx0 = sid * N_SPATIAL + (int(l0) | (int(u0) << 2) | (int(n0) << 4))
            s0   = int(px & np.uint8(3))
            f0  = all_sf[0, ctx0, s0];  cf0 = all_cf[0, ctx0, s0]
            m0  = magic[0, ctx0, s0]
            while st0 >= l_max * f0:
                out[ptr] = np.uint8(st0 & np.uint64(0xFF)); ptr += 1
                st0 >>= np.uint64(8)
            q0 = _mul_hi(st0, m0); r0_ = st0 - q0 * f0
            if r0_ >= f0: q0 += np.uint64(1); r0_ -= f0
            st0 = (q0 << m_bits) + r0_ + cf0

    states = np.zeros(4, dtype=np.uint64)
    states[0], states[1], states[2], states[3] = st0, st1, st2, st3
    return states, out[:ptr]


# ---------------------------------------------------------------------------
# Decoder kernel
# ---------------------------------------------------------------------------
@njit(cache=True, boundscheck=False, nogil=True)
def _rans_decode_sharded(bitstream:  npt.NDArray[np.uint8],
                         st0: uint64, st1: uint64, st2: uint64, st3: uint64,
                         h: int, w: int,
                         all_cf:    npt.NDArray[np.uint64],
                         all_sf:    npt.NDArray[np.uint64],
                         shard_map: npt.NDArray[np.uint8],
                         nsid: int) -> npt.NDArray[np.uint8]:
    """
    Forward-scan decoder.
    Maintains orig[h,w] alongside resid[h,w] so shard IDs are computable
    from already-reconstructed original pixel neighbors.
    Returns resid_2d (ZigZag residuals) for downstream reconstruct_2d_channels.
    """
    # Zero-padded internal arrays: border stays 0, loop from 1..h+1 / 1..w+1
    resid = np.zeros((h + 2, w + 2), dtype=np.uint8)
    orig  = np.zeros((h + 2, w + 2), dtype=np.uint8)

    l_lower = np.uint64(1 << 31)
    m_bits  = np.uint64(12)
    mask    = np.uint64((1 << 12) - 1)
    ptr     = len(bitstream) - 1

    for pi in range(1, h + 1):
        for pj in range(1, w + 1):
            # ----- shard ID from reconstructed original neighbors -----
            ag = orig[pi, pj-1]
            bg = orig[pi-1, pj]
            cg = orig[pi-1, pj-1]
            p_g = predict_med_standard(ag, bg, cg)
            sid = int(get_context_id_fast(ag, bg, cg, p_g, shard_map, nsid))

            # ----- bitplane spatial contexts from decoded residual neighbors -----
            r_l = resid[pi, pj-1]
            r_u = resid[pi-1, pj]
            r_n = resid[pi-1, pj-1]

            # ----- Decode Layer 0 -----
            l0 = r_l & np.uint8(3); u0 = r_u & np.uint8(3); n0 = r_n & np.uint8(3)
            ctx0 = sid * N_SPATIAL + (int(l0) | (int(u0) << 2) | (int(n0) << 4))
            cf0 = all_cf[0, ctx0]; sf0 = all_sf[0, ctx0]
            slot0 = st0 & mask
            sym0 = np.uint8(int(slot0 >= cf0[1]) + int(slot0 >= cf0[2]) + int(slot0 >= cf0[3]))
            st0 = sf0[sym0] * (st0 >> m_bits) + (slot0 - cf0[sym0])
            if st0 < l_lower and ptr >= 0:
                st0 = (st0 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1
                if st0 < l_lower and ptr >= 0:
                    st0 = (st0 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1

            # ----- Decode Layer 1 -----
            l1 = (r_l >> np.uint8(2)) & np.uint8(3)
            u1 = (r_u >> np.uint8(2)) & np.uint8(3)
            n1 = (r_n >> np.uint8(2)) & np.uint8(3)
            ctx1 = sid * N_SPATIAL + (int(l1) | (int(u1) << 2) | (int(n1) << 4))
            cf1 = all_cf[1, ctx1]; sf1 = all_sf[1, ctx1]
            slot1 = st1 & mask
            sym1 = np.uint8(int(slot1 >= cf1[1]) + int(slot1 >= cf1[2]) + int(slot1 >= cf1[3]))
            st1 = sf1[sym1] * (st1 >> m_bits) + (slot1 - cf1[sym1])
            if st1 < l_lower and ptr >= 0:
                st1 = (st1 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1
                if st1 < l_lower and ptr >= 0:
                    st1 = (st1 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1

            # ----- Decode Layer 2 -----
            l2 = (r_l >> np.uint8(4)) & np.uint8(3)
            u2 = (r_u >> np.uint8(4)) & np.uint8(3)
            n2 = (r_n >> np.uint8(4)) & np.uint8(3)
            ctx2 = sid * N_SPATIAL + (int(l2) | (int(u2) << 2) | (int(n2) << 4))
            cf2 = all_cf[2, ctx2]; sf2 = all_sf[2, ctx2]
            slot2 = st2 & mask
            sym2 = np.uint8(int(slot2 >= cf2[1]) + int(slot2 >= cf2[2]) + int(slot2 >= cf2[3]))
            st2 = sf2[sym2] * (st2 >> m_bits) + (slot2 - cf2[sym2])
            if st2 < l_lower and ptr >= 0:
                st2 = (st2 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1
                if st2 < l_lower and ptr >= 0:
                    st2 = (st2 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1

            # ----- Decode Layer 3 -----
            l3 = (r_l >> np.uint8(6)) & np.uint8(3)
            u3 = (r_u >> np.uint8(6)) & np.uint8(3)
            n3 = (r_n >> np.uint8(6)) & np.uint8(3)
            ctx3 = sid * N_SPATIAL + (int(l3) | (int(u3) << 2) | (int(n3) << 4))
            cf3 = all_cf[3, ctx3]; sf3 = all_sf[3, ctx3]
            slot3 = st3 & mask
            sym3 = np.uint8(int(slot3 >= cf3[1]) + int(slot3 >= cf3[2]) + int(slot3 >= cf3[3]))
            st3 = sf3[sym3] * (st3 >> m_bits) + (slot3 - cf3[sym3])
            if st3 < l_lower and ptr >= 0:
                st3 = (st3 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1
                if st3 < l_lower and ptr >= 0:
                    st3 = (st3 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1

            # ----- Assemble residual and update orig -----
            px = np.uint8(int(sym0) | (int(sym1) << 2) | (int(sym2) << 4) | (int(sym3) << 6))
            resid[pi, pj] = px
            orig[pi, pj]  = np.uint8((from_zigzag(px) + int(p_g)) & 0xFF)

    return resid[1:h+1, 1:w+1]


# ---------------------------------------------------------------------------
# Decoder for Rd/Bd channels (shard context from external reference channel)
# ---------------------------------------------------------------------------
@njit(cache=True, boundscheck=False, nogil=True)
def _rans_decode_sharded_with_ref(bitstream:  npt.NDArray[np.uint8],
                                   st0: uint64, st1: uint64, st2: uint64, st3: uint64,
                                   h: int, w: int,
                                   all_cf:    npt.NDArray[np.uint64],
                                   all_sf:    npt.NDArray[np.uint64],
                                   ref_ch:    npt.NDArray[np.uint8],
                                   shard_map: npt.NDArray[np.uint8],
                                   nsid: int) -> npt.NDArray[np.uint8]:
    """
    Forward-scan decoder for Rd/Bd channels.
    Shard_id is derived from ref_ch (the already-decoded green channel) so that
    encode and decode contexts are identical without any self-referential tracking.
    Returns resid_2d (ZigZag residuals) - caller passes to reconstruct_2d_channels.
    """
    # ref_ch arrives pre-padded (h+2, w+2); resid padded for zero-border neighbor reads
    resid = np.zeros((h + 2, w + 2), dtype=np.uint8)

    l_lower = np.uint64(1 << 31)
    m_bits  = np.uint64(12)
    mask    = np.uint64((1 << 12) - 1)
    ptr     = len(bitstream) - 1

    for pi in range(1, h + 1):
        for pj in range(1, w + 1):
            # ----- shard ID from decoded green reference -----
            ag = ref_ch[pi, pj-1]
            bg = ref_ch[pi-1, pj]
            cg = ref_ch[pi-1, pj-1]
            p_g = predict_med_standard(ag, bg, cg)
            sid = int(get_context_id_fast(ag, bg, cg, p_g, shard_map, nsid))

            # ----- bitplane spatial contexts from own decoded residual neighbors -----
            r_l = resid[pi, pj-1]
            r_u = resid[pi-1, pj]
            r_n = resid[pi-1, pj-1]

            # ----- Decode Layer 0 -----
            l0 = r_l & np.uint8(3); u0 = r_u & np.uint8(3); n0 = r_n & np.uint8(3)
            ctx0 = sid * N_SPATIAL + (int(l0) | (int(u0) << 2) | (int(n0) << 4))
            cf0 = all_cf[0, ctx0]; sf0 = all_sf[0, ctx0]
            slot0 = st0 & mask
            sym0 = np.uint8(int(slot0 >= cf0[1]) + int(slot0 >= cf0[2]) + int(slot0 >= cf0[3]))
            st0 = sf0[sym0] * (st0 >> m_bits) + (slot0 - cf0[sym0])
            if st0 < l_lower and ptr >= 0:
                st0 = (st0 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1
                if st0 < l_lower and ptr >= 0:
                    st0 = (st0 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1

            # ----- Decode Layer 1 -----
            l1 = (r_l >> np.uint8(2)) & np.uint8(3)
            u1 = (r_u >> np.uint8(2)) & np.uint8(3)
            n1 = (r_n >> np.uint8(2)) & np.uint8(3)
            ctx1 = sid * N_SPATIAL + (int(l1) | (int(u1) << 2) | (int(n1) << 4))
            cf1 = all_cf[1, ctx1]; sf1 = all_sf[1, ctx1]
            slot1 = st1 & mask
            sym1 = np.uint8(int(slot1 >= cf1[1]) + int(slot1 >= cf1[2]) + int(slot1 >= cf1[3]))
            st1 = sf1[sym1] * (st1 >> m_bits) + (slot1 - cf1[sym1])
            if st1 < l_lower and ptr >= 0:
                st1 = (st1 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1
                if st1 < l_lower and ptr >= 0:
                    st1 = (st1 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1

            # ----- Decode Layer 2 -----
            l2 = (r_l >> np.uint8(4)) & np.uint8(3)
            u2 = (r_u >> np.uint8(4)) & np.uint8(3)
            n2 = (r_n >> np.uint8(4)) & np.uint8(3)
            ctx2 = sid * N_SPATIAL + (int(l2) | (int(u2) << 2) | (int(n2) << 4))
            cf2 = all_cf[2, ctx2]; sf2 = all_sf[2, ctx2]
            slot2 = st2 & mask
            sym2 = np.uint8(int(slot2 >= cf2[1]) + int(slot2 >= cf2[2]) + int(slot2 >= cf2[3]))
            st2 = sf2[sym2] * (st2 >> m_bits) + (slot2 - cf2[sym2])
            if st2 < l_lower and ptr >= 0:
                st2 = (st2 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1
                if st2 < l_lower and ptr >= 0:
                    st2 = (st2 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1

            # ----- Decode Layer 3 -----
            l3 = (r_l >> np.uint8(6)) & np.uint8(3)
            u3 = (r_u >> np.uint8(6)) & np.uint8(3)
            n3 = (r_n >> np.uint8(6)) & np.uint8(3)
            ctx3 = sid * N_SPATIAL + (int(l3) | (int(u3) << 2) | (int(n3) << 4))
            cf3 = all_cf[3, ctx3]; sf3 = all_sf[3, ctx3]
            slot3 = st3 & mask
            sym3 = np.uint8(int(slot3 >= cf3[1]) + int(slot3 >= cf3[2]) + int(slot3 >= cf3[3]))
            st3 = sf3[sym3] * (st3 >> m_bits) + (slot3 - cf3[sym3])
            if st3 < l_lower and ptr >= 0:
                st3 = (st3 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1
                if st3 < l_lower and ptr >= 0:
                    st3 = (st3 << np.uint64(8)) | np.uint64(bitstream[ptr]); ptr -= 1

            resid[pi, pj] = np.uint8(int(sym0) | (int(sym1) << 2) | (int(sym2) << 4) | (int(sym3) << 6))

    return resid[1:h+1, 1:w+1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compress_bitplane_gray_sharded(gray_ch:   npt.NDArray[np.uint8],
                                   resid_2d:  npt.NDArray[np.uint8],
                                   shard_map: npt.NDArray[np.uint8],
                                   nsid: int) -> bytes:
    """
    Primary orchestrator: builds shard-conditioned PDF tables, runs the
    reverse-scan rANS encoder, then serialises to bytes.
    """
    n_shards = int(shard_map.max()) + 1 if nsid < 0 else nsid + 1
    n_ctx = n_shards * N_SPATIAL
    gray_ch_p  = np.pad(gray_ch,  1, constant_values=0)
    resid_2d_p = np.pad(resid_2d, 1, constant_values=0)
    f, cf = _build_pdf_sharded(resid_2d_p, gray_ch_p, shard_map, nsid, n_ctx, get_num_threads())

    states, bitstream = _rans_encode_sharded(resid_2d_p, gray_ch_p, cf, f, shard_map, nsid, n_ctx)

    # Compress frequency tables with Zstd (many uniform/zero rows - high ratio)
    tables_raw = f.astype(np.uint16).tobytes()
    tables_zstd = zstd.ZstdCompressor(level=1).compress(tables_raw)

    out = bytearray()
    out.append(BITPLANE_MAGIC)
    out.extend(np.array([len(tables_zstd)], dtype=np.uint32).tobytes())
    out.extend(tables_zstd)
    out.extend(states.tobytes())                                         # 32 bytes
    out.extend(np.array([len(bitstream)], dtype=np.uint32).tobytes())
    out.extend(bitstream.tobytes())
    return bytes(out)


def decompress_bitplane_gray_sharded(payload:   bytes,
                                     h: int, w: int,
                                     shard_map: npt.NDArray[np.uint8],
                                     nsid: int) -> npt.NDArray[np.uint8]:
    """
    Restores the ZigZag residual array from a sharded bitplane payload.
    The caller passes the result to reconstruct_2d_channels for final MED
    reconstruction (same pipeline as the legacy bitplane path).
    """
    n_shards = int(shard_map.max()) + 1 if nsid < 0 else nsid + 1
    n_ctx = n_shards * N_SPATIAL

    raw = np.frombuffer(payload, dtype=np.uint8)
    ptr = 0

    assert raw[ptr] == BITPLANE_MAGIC, "Not a sharded bitplane payload"
    ptr += 1

    tables_len = int(np.frombuffer(raw[ptr:ptr+4], dtype=np.uint32)[0])
    ptr += 4
    tables_raw = zstd.ZstdDecompressor().decompress(raw[ptr:ptr+tables_len].tobytes())
    ptr += tables_len

    f = np.frombuffer(tables_raw, dtype=np.uint16).reshape((4, n_ctx, 4)).astype(np.uint64)

    # Build cumulative frequencies (vectorised)
    cf = np.zeros((4, n_ctx, 5), dtype=np.uint64)
    cf[:, :, 1:] = np.cumsum(f, axis=2)

    states = np.frombuffer(raw[ptr:ptr+32], dtype=np.uint64)
    ptr += 32

    bs_len = int(np.frombuffer(raw[ptr:ptr+4], dtype=np.uint32)[0])
    ptr += 4
    bitstream = raw[ptr:ptr+bs_len]

    return _rans_decode_sharded(
        bitstream,
        states[0], states[1], states[2], states[3],
        h, w, cf, f, shard_map, nsid
    )


def compress_bitplane_rgb_sharded(gr_ch:    npt.NDArray[np.uint8],
                                   rd_ch:    npt.NDArray[np.uint8],
                                   bd_ch:    npt.NDArray[np.uint8],
                                   gr_resid: npt.NDArray[np.uint8],
                                   rd_resid: npt.NDArray[np.uint8],
                                   bd_resid: npt.NDArray[np.uint8],
                                   shard_map: npt.NDArray[np.uint8],
                                   nsid: int) -> bytes:
    """
    RGB sharded bitplane encoder.
    All three channels use the green channel (gr_ch) for shard context, matching
    shard_rgb.py's convention. All three channels are encoded in parallel via
    threads (Numba kernels release the GIL via nogil=True). Bitstream format:
      [0xFF]
      [gr_channel_block]  - tables_zstd_len:u32, tables_zstd, states:4*u64, bs_len:u32, bitstream
      [rd_channel_block]  - same structure
      [bd_channel_block]  - same structure
    """
    n_shards = int(shard_map.max()) + 1 if nsid < 0 else nsid + 1
    n_ctx = n_shards * N_SPATIAL

    def _pack_channel(resid: npt.NDArray[np.uint8],
                      ref:   npt.NDArray[np.uint8]) -> bytes:
        resid_p = np.pad(resid, 1, constant_values=0)
        ref_p   = np.pad(ref,   1, constant_values=0)
        f_nc, cf_nc = _build_pdf_sharded(resid_p, ref_p, shard_map, nsid, n_ctx, get_num_threads())
        states, bs = _rans_encode_sharded(resid_p, ref_p, cf_nc, f_nc, shard_map, nsid, n_ctx)
        tables_zstd = zstd.ZstdCompressor(level=1).compress(f_nc.astype(np.uint16).tobytes())
        blk = bytearray()
        blk.extend(np.array([len(tables_zstd)], dtype=np.uint32).tobytes())
        blk.extend(tables_zstd)
        blk.extend(states.tobytes())
        blk.extend(np.array([len(bs)], dtype=np.uint32).tobytes())
        blk.extend(bs.tobytes())
        return bytes(blk)

    # All three channels read gr_ch (never write it) → no data hazard.
    # Numba kernels have nogil=True so they run in true parallel threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fut_gr = ex.submit(_pack_channel, gr_resid, gr_ch)
        fut_rd = ex.submit(_pack_channel, rd_resid, gr_ch)
        fut_bd = ex.submit(_pack_channel, bd_resid, gr_ch)
        blk_gr = fut_gr.result()
        blk_rd = fut_rd.result()
        blk_bd = fut_bd.result()

    out = bytearray()
    out.append(BITPLANE_MAGIC)
    out.extend(blk_gr)
    out.extend(blk_rd)
    out.extend(blk_bd)
    return bytes(out)


def decompress_bitplane_rgb_sharded(payload:   bytes,
                                     h: int, w: int,
                                     shard_map: npt.NDArray[np.uint8],
                                     nsid: int):
    """
    RGB sharded bitplane decoder.
    Decodes green first (self-referential), then Rd and Bd in parallel using
    the reconstructed green channel for shard context.
    Returns (gr_rec, rd_rec, bd_rec) as (h, w) uint8 arrays ready for restore_channels.
    """
    from .transform import reconstruct_2d_channels

    n_shards = int(shard_map.max()) + 1 if nsid < 0 else nsid + 1
    n_ctx = n_shards * N_SPATIAL

    raw = np.frombuffer(payload, dtype=np.uint8)
    assert raw[0] == BITPLANE_MAGIC, "Not a sharded bitplane payload"
    ptr = 1

    decomp = zstd.ZstdDecompressor()

    def _unpack_channel(ptr):
        tables_len = int(np.frombuffer(raw[ptr:ptr+4], dtype=np.uint32)[0]); ptr += 4
        f = np.frombuffer(
            decomp.decompress(raw[ptr:ptr+tables_len].tobytes()),
            dtype=np.uint16
        ).reshape((4, n_ctx, 4)).astype(np.uint64)
        ptr += tables_len
        cf = np.zeros((4, n_ctx, 5), dtype=np.uint64)
        cf[:, :, 1:] = np.cumsum(f, axis=2)
        states = np.frombuffer(raw[ptr:ptr+32], dtype=np.uint64).copy(); ptr += 32
        bs_len = int(np.frombuffer(raw[ptr:ptr+4], dtype=np.uint32)[0]); ptr += 4
        bs = raw[ptr:ptr+bs_len]; ptr += bs_len
        return f, cf, states, bs, ptr

    # Unpack all three channel headers before spawning threads
    f_gr, cf_gr, st_gr, bs_gr, ptr = _unpack_channel(ptr)
    f_rd, cf_rd, st_rd, bs_rd, ptr = _unpack_channel(ptr)
    f_bd, cf_bd, st_bd, bs_bd, ptr = _unpack_channel(ptr)

    # ---- Green must be decoded first (shard context is self-referential) ----
    gr_resid = _rans_decode_sharded(
        bs_gr, st_gr[0], st_gr[1], st_gr[2], st_gr[3],
        h, w, cf_gr, f_gr, shard_map, nsid
    )
    gr_rec   = reconstruct_2d_channels(h, w, gr_resid)
    gr_rec_p = np.pad(gr_rec, 1, constant_values=0)

    # ---- Rd and Bd both read gr_rec_p (never write it) → parallel ----
    def _decode_rd():
        resid = _rans_decode_sharded_with_ref(
            bs_rd, st_rd[0], st_rd[1], st_rd[2], st_rd[3],
            h, w, cf_rd, f_rd, gr_rec_p, shard_map, nsid
        )
        return reconstruct_2d_channels(h, w, resid)

    def _decode_bd():
        resid = _rans_decode_sharded_with_ref(
            bs_bd, st_bd[0], st_bd[1], st_bd[2], st_bd[3],
            h, w, cf_bd, f_bd, gr_rec_p, shard_map, nsid
        )
        return reconstruct_2d_channels(h, w, resid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_rd = ex.submit(_decode_rd)
        fut_bd = ex.submit(_decode_bd)
        rd_rec = fut_rd.result()
        bd_rec = fut_bd.result()

    return gr_rec, rd_rec, bd_rec
