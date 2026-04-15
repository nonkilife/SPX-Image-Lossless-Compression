"""
ZPNG-CSDE v6.2 [Flexible-Shard Architecture]
Module: zpng_codec
Role: Bitstream Orchestration.
Description: Logic for packing and unpacking the ZPNG tiered bitstream container.
Architecture: Structured serialization layer bridging the Model and rANS pillars.
"""

import numpy as np
import numpy.typing as npt
import zstandard as zstd
from numba import njit, prange, uint8, uint16, uint32, uint64
from typing import Tuple, List, Optional
from .common import (
    TOTAL_SHARDS, FLAG_RGBA, FLAG_SIMPLE, FLAG_RAW, FLAG_GRAYSCALE, FLAG_COLOR_GSUB, FLAG_BITPLANE,
    PROFILE_RGB
)
from . import rans_bitplane as bit_rans
from .rans import (
    rans_encode_shards_parallel, 
    build_pdf_tables_from_shards, L_LOWER,
    compact_pdf_tables
)

def pack_bitstream_v5(h: int, w: int, is_rgba: bool, is_grayscale: bool, use_gsub: bool,
                     shard_counts: npt.NDArray[np.uint32], shard_offsets_p1: npt.NDArray[np.uint32],
                     shard_widths: npt.NDArray[np.uint16],
                     shard_medians: npt.NDArray[np.uint8],
                     res_flat: npt.NDArray[np.uint8],
                     res_a: npt.NDArray[np.uint8],
                     metadata_bytes: bytes) -> Tuple[bytes, npt.NDArray[np.uint8]]:
    """
    Serializes compressed data into the final ZPNG file block.
    
    Bitstream Architecture (v6.5):
    [0-15]     Global Header (Height, Width, Metadata Length, Bit Flags)
    [16-N]     SRB Chunk: Widths (n_shards per channel), Medians, Modes
    [N-M]      PDF Chunk (Zstd): Compacted frequency tables for Dynamic Modes (0, 1)
    [M-O]      Shard Block: Sizes for each active shard, followed by rANS byte payloads
    [O-END]    Optional Metadata
    """
    flag: int = int(is_rgba)
    if is_grayscale: flag |= FLAG_GRAYSCALE
    if use_gsub: flag |= FLAG_COLOR_GSUB

    metadata_len = len(metadata_bytes)
    header_base: bytes = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
    
    profile = PROFILE_RGB
    n_shards = profile.total_shards

    # Widths are stored as uint8 using mod-256 encoding: value 0 represents 256.
    # The decoder (decompress.py) restores this via: np.where(r == 0, 256, r).
    if is_grayscale:
        header_widths: bytes = (shard_widths[0, :n_shards] % 256).astype(np.uint8).tobytes()
        header_medians: bytes = shard_medians[0, :n_shards].tobytes()
    else:
        header_widths: bytes = (shard_widths[:, :n_shards] % 256).astype(np.uint8).tobytes()
        header_medians: bytes = shard_medians[:, :n_shards].tobytes()
    
    # 1. ANS Frequency Modeling & Mode Selection
    sharded_payload: bytearray = bytearray()
    
    gr_len = int(np.sum(shard_counts[0]))
    shard_gr = res_flat[0:gr_len]
    gr_shards: List[npt.NDArray[np.uint8]] = [shard_gr[shard_offsets_p1[0,s]:shard_offsets_p1[0,s]+shard_counts[0,s]] for s in range(n_shards)]
    
    if not is_grayscale:
        rd_len = int(np.sum(shard_counts[1]))
        shard_rd = res_flat[gr_len : gr_len + rd_len]
        shard_bd = res_flat[gr_len + rd_len :]
        rd_shards: List[npt.NDArray[np.uint8]] = [shard_rd[shard_offsets_p1[1,s]:shard_offsets_p1[1,s]+shard_counts[1,s]] for s in range(n_shards)]
        bd_shards: List[npt.NDArray[np.uint8]] = [shard_bd[shard_offsets_p1[2,s]:shard_offsets_p1[2,s]+shard_counts[2,s]] for s in range(n_shards)]
        gr_cum, gr_sym, gr_modes = build_pdf_tables_from_shards(gr_shards, shard_widths[0])
        rd_cum, rd_sym, rd_modes = build_pdf_tables_from_shards(rd_shards, shard_widths[1])
        bd_cum, bd_sym, bd_modes = build_pdf_tables_from_shards(bd_shards, shard_widths[2])
    else:
        gr_cum, gr_sym, gr_modes = build_pdf_tables_from_shards(gr_shards, shard_widths[0])
    
    if is_grayscale:
        all_sym_freqs_flat = gr_sym
        all_widths_flat = shard_widths[0]
        header_modes: bytes = gr_modes.tobytes()
        all_modes_internal = gr_modes
    else:
        all_sym_freqs_flat = np.concatenate((gr_sym, rd_sym, bd_sym))
        all_widths_flat = np.concatenate((shard_widths[0, :n_shards], shard_widths[1, :n_shards], shard_widths[2, :n_shards]))
        header_modes: bytes = np.concatenate((gr_modes, rd_modes, bd_modes)).tobytes()
        all_modes_internal = np.concatenate((gr_modes, rd_modes, bd_modes))
    
    header: bytes = header_base + header_widths + header_medians + header_modes
    
    pdf_compact_block = compact_pdf_tables(all_sym_freqs_flat, all_widths_flat, all_modes_internal)
    
    # [Internal reuse of zstd decompressor/compressor is handled by callers or locally]
    c_pdf = zstd.ZstdCompressor(level=1).compress(pdf_compact_block.tobytes())
    
    # 1a. Write PDF block
    sharded_payload += np.array([len(c_pdf)], dtype='<u4').tobytes() + c_pdf
    
    # 1b. Write Shard Counts
    if is_grayscale:
        sharded_payload += np.array([shard_counts[0].nbytes], dtype='<u4').tobytes() + shard_counts[0].tobytes()
    else:
        sharded_payload += np.array([shard_counts.nbytes], dtype='<u4').tobytes() + shard_counts.tobytes()
    
    # 1d. rANS Encode Shards
    if is_grayscale:
        shard_lengths_ans: npt.NDArray[np.uint32] = shard_counts[0].ravel().astype(np.uint32)
        all_cum_stack: npt.NDArray[np.uint64] = gr_cum
        all_sym_stack: npt.NDArray[np.uint64] = gr_sym
    else:
        shard_lengths_ans: npt.NDArray[np.uint32] = shard_counts.ravel().astype(np.uint32)
        all_cum_stack: npt.NDArray[np.uint64] = np.concatenate((gr_cum, rd_cum, bd_cum))
        all_sym_stack: npt.NDArray[np.uint64] = np.concatenate((gr_sym, rd_sym, bd_sym))

    shard_offsets_ans: npt.NDArray[np.uint32] = np.zeros(len(shard_lengths_ans), dtype=np.uint32)
    if len(shard_lengths_ans) > 0:
        shard_offsets_ans[1:] = np.cumsum(shard_lengths_ans[:-1])
        
    final_states, bitstreams_flat, bs_offsets, bs_lengths = rans_encode_shards_parallel(
        res_flat, shard_offsets_ans, shard_lengths_ans,
        all_cum_stack, all_sym_stack, L_LOWER
    )
    
    for idx in range(len(shard_lengths_ans)):
        sharded_payload += final_states[idx].astype('<u8').tobytes()
        sharded_payload += np.array([int(bs_lengths[idx])], dtype='<u4').tobytes()
        if bs_lengths[idx] > 0:
            off = int(bs_offsets[idx])
            sharded_payload += bitstreams_flat[off:off+int(bs_lengths[idx])].tobytes()

    # 1d. Encode Alpha (Traditional Zstd)
    if is_rgba:
        c_alpha = zstd.ZstdCompressor(level=1).compress(res_a.tobytes())
        sharded_payload += np.array([len(c_alpha)], dtype='<u4').tobytes() + c_alpha

    # 1e. Return full modes for diagnostics
    if is_grayscale:
        # Re-broadcast for consistency
        modes_diag = np.zeros((3, n_shards), dtype=np.uint8)
        modes_diag[0] = gr_modes
        return b"ZPNGCSDE" + header + bytes(sharded_payload) + metadata_bytes, modes_diag
    else:
        modes_diag = np.zeros((3, n_shards), dtype=np.uint8)
        modes_diag[0], modes_diag[1], modes_diag[2] = gr_modes, rd_modes, bd_modes
        return b"ZPNGCSDE" + header + bytes(sharded_payload) + metadata_bytes, modes_diag

def pack_bitplane_bitstream_v5(h: int, w: int, is_rgba: bool, is_grayscale: bool, use_gsub: bool,
                             resid_gr: npt.NDArray[np.uint8], metadata_bytes: bytes) -> bytes:
    """ Bitplane-Contextual packing path (v5.3). Currently optimized for Grayscale. """
    flag: int = FLAG_BITPLANE
    if is_rgba: flag |= FLAG_RGBA
    if is_grayscale: flag |= FLAG_GRAYSCALE
    if use_gsub: flag |= FLAG_COLOR_GSUB
    
    metadata_len = len(metadata_bytes)
    # Header: h, w, metadata_len, flag (4 x 4 bytes = 16 bytes)
    header: bytes = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
    
    # 2D Bit-Context Encoding (Greyscale path)
    if is_grayscale:
        bit_payload = bit_rans.compress_bitplane_gray(resid_gr)
    else:
        # Placeholder for color bitplane (will follow same logic for each channel)
        raise NotImplementedError("Color Bitplane mode not yet implemented.")
        
    return b"ZPNGCSDE" + header + bit_payload + metadata_bytes

def unpack_bitstream_v5(compressed_data: bytes, h: int, w: int, is_rgba: bool, is_grayscale: bool,
                       shard_widths: npt.NDArray[np.uint16], shard_modes: npt.NDArray[np.uint8],
                       flag: int, metadata_len: int) -> Tuple:
    """
    Deserializes the ZPNG bitstream format back into parsed frequency tables and residuals.

    Processing Steps:
    1. Parses the Shard Metadata block (Widths, Medians, Modes) dynamically.
    2. Uses Zstd to de-compact the internal PDF dictionary arrays.
    3. Interprets the Shard Lengths matrix to carve up the rANS payloads array.
    4. Triggers the 4-Way SIMD rANS parallel decoding block for all channels.
    """
    def read_block_meta(buf: bytes, pos: int) -> Tuple[int, int, int]:
        if pos + 4 > len(buf):
            raise ValueError("Unexpected End of File: Missing block metadata.")
        b_len_v: int = int(np.frombuffer(buf[pos:pos+4], dtype='<u4')[0])
        if pos + 4 + b_len_v > len(buf):
            raise ValueError(f"Truncated bitstream: Expected {b_len_v} bytes, got {len(buf) - (pos + 4)}")
        return b_len_v, pos + 4, pos + 4 + b_len_v

    from .rans import (
        rans_decode_shards_parallel, build_all_lookups,
        expand_pdf_tables
    )
    dctx = zstd.ZstdDecompressor()

    if flag & FLAG_BITPLANE:
        if is_grayscale:
            gray_res = bit_rans.decompress_bitplane_gray(compressed_data, h, w).flatten()
            return gray_res, None, None, None, None, None, None, None
        else:
            raise NotImplementedError("Color Bitplane unpacking not yet implemented.")

    # [v6.6] Determine Shard Count
    profile = PROFILE_RGB
    n_shards = profile.total_shards
    n_colors = 1 if is_grayscale else 3
    
    p: int = 0
    # 1. Read compacted PDF frequencies
    pdf_len, pdf_start, p = read_block_meta(compressed_data, p)
    pdf_raw: npt.NDArray[np.uint8] = np.frombuffer(dctx.decompress(compressed_data[pdf_start:p]), dtype=np.uint8)
    
    if is_grayscale:
        widths_flat: npt.NDArray[np.uint16] = shard_widths[0, :n_shards].flatten()
        modes_flat: npt.NDArray[np.uint8] = shard_modes[0, :n_shards].flatten()
        all_sym_freqs_flat: npt.NDArray[np.uint64] = expand_pdf_tables(pdf_raw, widths_flat, modes_flat)
        all_sym_freqs = np.zeros((3, n_shards, 256), dtype=np.uint64)
        all_sym_freqs[0] = all_sym_freqs_flat.reshape((n_shards, 256))
    else:
        widths_flat: npt.NDArray[np.uint16] = shard_widths[:, :n_shards].flatten()
        modes_flat: npt.NDArray[np.uint8] = shard_modes[:, :n_shards].flatten()
        all_sym_freqs_flat: npt.NDArray[np.uint64] = expand_pdf_tables(pdf_raw, widths_flat, modes_flat)
        all_sym_freqs = all_sym_freqs_flat.reshape((3, n_shards, 256))
    
    all_cum_freqs: npt.NDArray[np.uint64] = np.zeros((3, n_shards, 257), dtype=np.uint64)
    all_cum_freqs[:, :, 1:] = np.cumsum(all_sym_freqs, axis=2)

    # 2. Read Shard Counts
    sc_len, sc_start, p = read_block_meta(compressed_data, p)
    sc_block: npt.NDArray[np.uint32] = np.frombuffer(compressed_data[sc_start:p], dtype='<u4')
    
    shard_counts: npt.NDArray[np.uint32] = np.zeros((3, n_shards), dtype=np.uint32)
    if is_grayscale:
        shard_counts[0] = sc_block.copy().reshape((n_shards,))
    else:
        shard_counts = sc_block.copy().reshape((3, n_shards))

    # 4. Parallel rANS Decoding (v7.2 [HIGH-THROUGHPUT])
    all_lookups: npt.NDArray[np.uint8] = build_all_lookups(all_cum_freqs)
    
    # [v7.2] Flatten stacks to maximize multi-core thread saturation
    if is_grayscale:
        counts_stack = shard_counts[0]
        all_cum_stack = all_cum_freqs[0]
        all_sym_stack = all_sym_freqs[0]
        all_lookups_stack = all_lookups[0]
    else:
        counts_stack = shard_counts.flatten()
        all_cum_stack = all_cum_freqs.reshape((3 * n_shards, 257))
        all_sym_stack = all_sym_freqs.reshape((3 * n_shards, 256))
        all_lookups_stack = all_lookups.reshape((3 * n_shards, 4096))
        
    num_targets = len(counts_stack)
    total_res = int(np.sum(counts_stack))
    all_res_flat: npt.NDArray[np.uint8] = np.empty(total_res, dtype=np.uint8)
    
    out_offsets = np.zeros(num_targets, dtype=np.uint32)
    if num_targets > 0:
        out_offsets[1:] = np.cumsum(counts_stack[:-1], dtype=np.uint32)

    # Phase 1: I/O Pre-Scan (Sequential)
    states = np.zeros((num_targets, 4), dtype=np.uint64)
    bs_offsets = np.zeros(num_targets, dtype=np.uint32)
    bs_lengths = np.zeros(num_targets, dtype=np.uint32)
    
    for idx in range(num_targets):
        if p + 36 > len(compressed_data):
            raise ValueError(f"Truncated bitstream: shard {idx}/{num_targets} header missing at offset {p}")
            
        chunk_meta: npt.NDArray[np.uint32] = np.frombuffer(compressed_data[p:p+36], dtype='<u4')
        states[idx, 0] = np.uint64(chunk_meta[0]) | (np.uint64(chunk_meta[1]) << 32)
        states[idx, 1] = np.uint64(chunk_meta[2]) | (np.uint64(chunk_meta[3]) << 32)
        states[idx, 2] = np.uint64(chunk_meta[4]) | (np.uint64(chunk_meta[5]) << 32)
        states[idx, 3] = np.uint64(chunk_meta[6]) | (np.uint64(chunk_meta[7]) << 32)
        b_len = int(chunk_meta[8])
        p += 36
        
        bs_offsets[idx] = np.uint32(p)
        bs_lengths[idx] = np.uint32(b_len)
        p += b_len

    # Phase 2: Parallel Entropy Decoding Dispatch
    compressed_data_arr = np.frombuffer(compressed_data, dtype=np.uint8)
    rans_decode_shards_parallel(
        compressed_data_arr,
        states,
        bs_offsets,
        bs_lengths,
        all_cum_stack,
        all_sym_stack,
        all_lookups_stack,
        all_res_flat,
        out_offsets,
        counts_stack.astype(np.uint32)
    )

    # Phase 3: Zero-Copy Recombination
    if is_grayscale:
        res_gr_flat = all_res_flat
        res_rd_flat = np.empty(0, dtype=np.uint8)
        res_bd_flat = np.empty(0, dtype=np.uint8)
        gr_offs = out_offsets
        rd_offs = np.zeros(n_shards, dtype=np.uint32)
        bd_offs = np.zeros(n_shards, dtype=np.uint32)
    else:
        gr_len = int(np.sum(counts_stack[:n_shards]))
        rd_len = int(np.sum(counts_stack[n_shards:2*n_shards]))
        
        res_gr_flat = all_res_flat[:gr_len]
        res_rd_flat = all_res_flat[gr_len:gr_len+rd_len]
        res_bd_flat = all_res_flat[gr_len+rd_len:]
        
        gr_offs = out_offsets[:n_shards]
        rd_offs = out_offsets[n_shards:2*n_shards] - np.uint32(gr_len)
        bd_offs = out_offsets[2*n_shards:] - np.uint32(gr_len + rd_len)

    res_a_flat: Optional[npt.NDArray[np.uint8]] = None
    if is_rgba:
        a_len, a_start, p2 = read_block_meta(compressed_data, p)
        res_a_flat = np.frombuffer(dctx.decompress(compressed_data[a_start:p2]), dtype=np.uint8)
        
    return res_gr_flat, res_rd_flat, res_bd_flat, gr_offs, rd_offs, bd_offs, shard_counts, res_a_flat
