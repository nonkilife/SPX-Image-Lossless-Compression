"""
SPX v8.3.2 [Stable Parallel Architecture]
Module: rans_bitplane
Role: Entropy coding combining N-shard gradient context with
      2-bit spatial bitplane context.

Phase 2 Rust migration: all hot-path kernels replaced by spx_rans Rust extension.
Public API and bitstream format are unchanged.
"""

__version__ = "8.3.2"

import numpy as np
import numpy.typing as npt
from typing import Tuple
import concurrent.futures

from .predictor import from_zigzag
from .sharding import ShardProfile
from .transform import reconstruct_2d_channels
from .codec import get_zstd_comp, get_zstd_decomp

import spx_rans as _rs

# ---------------------------------------------------------------------------
# Constants (kept for any external references)
# ---------------------------------------------------------------------------
N_SPATIAL: int = 64
BITPLANE_MAGIC: int = 0xFF


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _luts(profile: ShardProfile):
    """Returns (s_lut_flat, i_lut, d_lut_flat) as contiguous u8 arrays."""
    return (
        np.ascontiguousarray(profile.spatial_lut.ravel(), dtype=np.uint8),
        np.ascontiguousarray(profile.intensity_lut, dtype=np.uint8),
        np.ascontiguousarray(profile.dispatch_lut.ravel(), dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress_bitplane_gray_sharded(h: int, w: int,
                                   gray_ch_p:  npt.NDArray[np.uint8],
                                   resid_2d_p: npt.NDArray[np.uint8],
                                   profile:    ShardProfile) -> bytes:
    n_ctx = profile.total_shards * N_SPATIAL
    sl, il, dl = _luts(profile)
    rf = np.ascontiguousarray(resid_2d_p.ravel(), dtype=np.uint8)
    gf = np.ascontiguousarray(gray_ch_p.ravel(), dtype=np.uint8)

    f, cf = _rs.bp_build_pdf_sharded(rf, gf, sl, il, dl, h, w, n_ctx, False)
    states, bitstream = _rs.bp_encode_sharded(rf, gf, cf, f, sl, il, dl, h, w, n_ctx, False)

    tables_zstd = get_zstd_comp(level=3).compress(f.tobytes())
    out = bytearray()
    out.append(BITPLANE_MAGIC)
    out.extend(np.array([len(tables_zstd)], dtype=np.uint32).tobytes())
    out.extend(tables_zstd)
    out.extend(states.astype(np.uint64).tobytes())
    out.extend(np.array([len(bitstream)], dtype=np.uint32).tobytes())
    out.extend(bitstream.tobytes())
    return bytes(out)


def decompress_bitplane_gray_sharded(payload:   bytes,
                                     h: int, w: int,
                                     profile:   ShardProfile) -> Tuple[npt.NDArray[np.uint8], int]:
    n_ctx = profile.total_shards * N_SPATIAL
    sl, il, dl = _luts(profile)

    raw = np.frombuffer(payload, dtype=np.uint8)
    ptr = 0
    if raw[ptr] != BITPLANE_MAGIC:
        raise ValueError("Not a sharded bitplane payload")
    ptr += 1

    tables_len = int(np.frombuffer(raw[ptr:ptr+4], dtype=np.uint32)[0])
    ptr += 4
    tables_raw = get_zstd_decomp().decompress(raw[ptr:ptr+tables_len].tobytes())
    ptr += tables_len

    f = np.frombuffer(tables_raw, dtype=np.uint16).reshape((4, n_ctx, 4)).copy()
    cf = np.zeros((4, n_ctx, 5), dtype=np.uint16)
    cf[:, :, 1:] = np.cumsum(f, axis=2)

    states = np.frombuffer(raw[ptr:ptr+32], dtype=np.uint64)
    ptr += 32
    bs_len = int(np.frombuffer(raw[ptr:ptr+4], dtype=np.uint32)[0])
    ptr += 4
    bitstream = np.ascontiguousarray(raw[ptr:ptr+bs_len], dtype=np.uint8)
    ptr += bs_len

    resid = _rs.bp_decode_sharded(
        bitstream, int(states[0]), int(states[1]), int(states[2]), int(states[3]),
        h, w, cf, f, sl, il, dl
    )
    return resid, ptr


def compress_bitplane_rgb_sharded(h: int, w: int,
                                   gr_ref_p: npt.NDArray[np.uint8],
                                   gr_p:     npt.NDArray[np.uint8],
                                   rd_p:     npt.NDArray[np.uint8],
                                   bd_p:     npt.NDArray[np.uint8],
                                   profile:  ShardProfile) -> bytes:
    n_ctx = profile.total_shards * N_SPATIAL
    sl, il, dl = _luts(profile)
    grf = np.ascontiguousarray(gr_p.ravel(), dtype=np.uint8)
    rdf = np.ascontiguousarray(rd_p.ravel(), dtype=np.uint8)
    bdf = np.ascontiguousarray(bd_p.ravel(), dtype=np.uint8)
    grrf = np.ascontiguousarray(gr_ref_p.ravel(), dtype=np.uint8)

    # Phase 1: fused 3-channel histogram
    f_gr, cf_gr, f_rd, cf_rd, f_bd, cf_bd = _rs.bp_build_pdf_sharded_rgb(
        grf, rdf, bdf, grrf, sl, il, dl, h, w, n_ctx
    )

    # Phase 2: encode three channels concurrently (Rust releases GIL)
    def _encode(resid_f, gray_f, cf, f, is_chroma):
        return _rs.bp_encode_sharded(resid_f, gray_f, cf, f, sl, il, dl, h, w, n_ctx, is_chroma)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fut_gr = ex.submit(_encode, grf, grrf, cf_gr, f_gr, False)
        fut_rd = ex.submit(_encode, rdf, grrf, cf_rd, f_rd, True)
        fut_bd = ex.submit(_encode, bdf, grrf, cf_bd, f_bd, True)
        states_gr, bs_gr = fut_gr.result()
        states_rd, bs_rd = fut_rd.result()
        states_bd, bs_bd = fut_bd.result()

    def _serialise(f, states, bs) -> bytes:
        tables_zstd = get_zstd_comp(level=3).compress(f.tobytes())
        blk = bytearray()
        blk.extend(np.array([len(tables_zstd)], dtype=np.uint32).tobytes())
        blk.extend(tables_zstd)
        blk.extend(states.astype(np.uint64).tobytes())
        blk.extend(np.array([len(bs)], dtype=np.uint32).tobytes())
        blk.extend(bs.tobytes())
        return bytes(blk)

    out = bytearray()
    out.append(BITPLANE_MAGIC)
    out.extend(_serialise(f_gr, states_gr, bs_gr))
    out.extend(_serialise(f_rd, states_rd, bs_rd))
    out.extend(_serialise(f_bd, states_bd, bs_bd))
    return bytes(out)


def decompress_bitplane_rgb_sharded(payload:   bytes,
                                     h: int, w: int,
                                     profile:   ShardProfile) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], int]:
    n_ctx = profile.total_shards * N_SPATIAL
    sl, il, dl = _luts(profile)

    raw = np.frombuffer(payload, dtype=np.uint8)
    if raw[0] != BITPLANE_MAGIC:
        raise ValueError("Not a sharded bitplane payload")
    ptr = 1

    def _unpack_channel(ptr):
        tables_len = int(np.frombuffer(raw[ptr:ptr+4], dtype=np.uint32)[0]); ptr += 4
        f = np.frombuffer(
            get_zstd_decomp().decompress(raw[ptr:ptr+tables_len].tobytes()),
            dtype=np.uint16
        ).reshape((4, n_ctx, 4)).copy()
        ptr += tables_len
        cf = np.zeros((4, n_ctx, 5), dtype=np.uint16)
        cf[:, :, 1:] = np.cumsum(f, axis=2)
        states = np.frombuffer(raw[ptr:ptr+32], dtype=np.uint64).copy(); ptr += 32
        bs_len = int(np.frombuffer(raw[ptr:ptr+4], dtype=np.uint32)[0]); ptr += 4
        bs = np.ascontiguousarray(raw[ptr:ptr+bs_len], dtype=np.uint8); ptr += bs_len
        return f, cf, states, bs, ptr

    f_gr, cf_gr, st_gr, bs_gr, ptr = _unpack_channel(ptr)
    f_rd, cf_rd, st_rd, bs_rd, ptr = _unpack_channel(ptr)
    f_bd, cf_bd, st_bd, bs_bd, ptr = _unpack_channel(ptr)

    # Green must be decoded first (self-referential shard context)
    gr_resid = _rs.bp_decode_sharded(
        bs_gr, int(st_gr[0]), int(st_gr[1]), int(st_gr[2]), int(st_gr[3]),
        h, w, cf_gr, f_gr, sl, il, dl
    )
    gr_rec   = reconstruct_2d_channels(h, w, gr_resid)
    gr_rec_p = np.pad(gr_rec, 1, constant_values=0)
    gr_ref_f = np.ascontiguousarray(gr_rec_p.ravel(), dtype=np.uint8)

    # Rd and Bd both read gr_rec_p — parallel decode
    def _decode_rd():
        resid = _rs.bp_decode_sharded_with_ref(
            bs_rd, int(st_rd[0]), int(st_rd[1]), int(st_rd[2]), int(st_rd[3]),
            h, w, cf_rd, f_rd, gr_ref_f, sl, il, dl
        )
        return reconstruct_2d_channels(h, w, resid)

    def _decode_bd():
        resid = _rs.bp_decode_sharded_with_ref(
            bs_bd, int(st_bd[0]), int(st_bd[1]), int(st_bd[2]), int(st_bd[3]),
            h, w, cf_bd, f_bd, gr_ref_f, sl, il, dl
        )
        return reconstruct_2d_channels(h, w, resid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_rd = ex.submit(_decode_rd)
        fut_bd = ex.submit(_decode_bd)
        rd_rec = fut_rd.result()
        bd_rec = fut_bd.result()

    return gr_rec, rd_rec, bd_rec, ptr
