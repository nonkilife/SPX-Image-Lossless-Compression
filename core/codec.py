"""
SPX [Stable Parallel Architecture]
Module: codec
Role: Bitstream Orchestration.
Description: Logic for packing and unpacking the SPX tiered bitstream container.
Architecture: Structured serialization layer bridging the Model and rANS pillars.
Engineering Rationale:
1. Deterministic Block Order: The header and SRB (Metadata) must appear first
   to define shard widths and medians, which are required for the decoder to
   pre-allocate the rANS state buffers.
2. Zero-Copy Parallelism: Shard payloads are stored with explicit byte-lengths
   preceding the content, allowing the decompressor to spawn independent threads
   that jump directly to their target payload without sequential bit-scanning.

Bitstream Structure:
```mermaid
graph TD
    Meta[Header: H, W, Flags] --> SRB[SRB Block: Widths, Medians, Modes]
    SRB --> PDF[PDF Block: Zstd Compacted Tables]
    PDF --> SC[Shard Counts Block]
    SC --> Data[Shard Block: rANS Payloads]
    Data --> Alpha[Alpha Block, if RGBA]
    Alpha --> End[Trailing Metadata]
```
"""

import numpy as np
import numpy.typing as npt
import zstandard as zstd
import concurrent.futures
from typing import Tuple, List, Optional, Union, BinaryIO
from .common import (
    FLAG_RGBA, FLAG_GRAYSCALE, FLAG_COLOR_GSUB, FLAG_BITPLANE
)
from .sharding import PROFILE_RGB
from .rans_bitplane import decompress_bitplane_gray_sharded
from .rans import (
    rans_encode_shards_parallel,
    build_pdf_tables_from_shards, L_LOWER,
    compact_pdf_tables,
    rans_decode_4way_core, build_all_lookups,
    expand_pdf_tables
)

def pack_bitstream(h: int, w: int, is_rgba: bool, is_grayscale: bool, use_gsub: bool,
                     shard_counts: npt.NDArray[np.uint32], shard_offsets_p1: npt.NDArray[np.uint32],
                     shard_widths: npt.NDArray[np.uint16],
                     res_flat: npt.NDArray[np.uint8],
                     res_a: npt.NDArray[np.uint8],
                     metadata_bytes: bytes) -> Tuple[bytes, npt.NDArray[np.uint8]]:
    """
    Serializes compressed data into the final SPX file block.
    
    Bitstream Architecture:
    [0-15]     Global Header (Height, Width, Metadata Length, Bit Flags)
    [16-N]     SRB Chunk: Widths (n_shards per channel), Medians, Modes
    [N-M]      PDF Chunk (Zstd): Compacted frequency tables for Dynamic Modes (0, 3)
    [M-O]      Shard Block: Sizes for each active shard, followed by rANS byte payloads
    [O-END]    Optional Metadata
    """
    flag: int = FLAG_RGBA if is_rgba else 0
    if is_grayscale: flag |= FLAG_GRAYSCALE
    if use_gsub: flag |= FLAG_COLOR_GSUB

    metadata_len = len(metadata_bytes)
    header_base: bytes = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
    
    profile = PROFILE_RGB
    n_shards = profile.total_shards

    # Widths are stored as uint8 using mod-256 encoding: value 0 represents 256.
    # The decoder (decompress.py) restores this via: np.where(r == 0, 256, r).
    # SRB Block: Widths and Modes
    if is_grayscale:
        header_widths: bytes = (shard_widths[0, :n_shards] % 256).astype(np.uint8).tobytes()
    else:
        header_widths: bytes = (shard_widths[:, :n_shards] % 256).astype(np.uint8).tobytes()
    
    sharded_payload: bytearray = bytearray()

    # Split res_flat into per-channel shard views
    gr_len = int(np.sum(shard_counts[0]))
    shard_gr = res_flat[0:gr_len]
    gr_shards: List[npt.NDArray[np.uint8]] = [shard_gr[shard_offsets_p1[0,s]:shard_offsets_p1[0,s]+shard_counts[0,s]] for s in range(n_shards)]

    if not is_grayscale:
        rd_len = int(np.sum(shard_counts[1]))
        shard_rd = res_flat[gr_len : gr_len + rd_len]
        shard_bd = res_flat[gr_len + rd_len :]
        rd_shards: List[npt.NDArray[np.uint8]] = [shard_rd[shard_offsets_p1[1,s]:shard_offsets_p1[1,s]+shard_counts[1,s]] for s in range(n_shards)]
        bd_shards: List[npt.NDArray[np.uint8]] = [shard_bd[shard_offsets_p1[2,s]:shard_offsets_p1[2,s]+shard_counts[2,s]] for s in range(n_shards)]

    # Build frequency tables (PDF modeling)
    gr_cum, gr_sym, gr_modes = build_pdf_tables_from_shards(gr_shards, shard_widths[0])
    if is_grayscale:
        all_sym_freqs_flat = gr_sym
        all_widths_flat = shard_widths[0, :n_shards]
        all_modes_internal = gr_modes
    else:
        rd_cum, rd_sym, rd_modes = build_pdf_tables_from_shards(rd_shards, shard_widths[1])
        bd_cum, bd_sym, bd_modes = build_pdf_tables_from_shards(bd_shards, shard_widths[2])
        all_sym_freqs_flat = np.concatenate((gr_sym, rd_sym, bd_sym))
        all_widths_flat = np.concatenate((shard_widths[0, :n_shards], shard_widths[1, :n_shards], shard_widths[2, :n_shards]))
        all_modes_internal = np.concatenate((gr_modes, rd_modes, bd_modes))
    header_modes: bytes = all_modes_internal.tobytes()
    
    header: bytes = header_base + header_widths + header_modes
    
    pdf_compact_block = compact_pdf_tables(all_sym_freqs_flat, all_widths_flat, all_modes_internal)
    
    c_pdf = zstd.ZstdCompressor(level=1).compress(pdf_compact_block.tobytes())

    # Step 1: Write PDF block
    sharded_payload += np.array([len(c_pdf)], dtype='<u4').tobytes() + c_pdf

    # Step 2: Write shard counts
    if is_grayscale:
        sharded_payload += np.array([shard_counts[0].nbytes], dtype='<u4').tobytes() + shard_counts[0].tobytes()
    else:
        sharded_payload += np.array([shard_counts.nbytes], dtype='<u4').tobytes() + shard_counts.tobytes()

    # Step 3: rANS encode shards
    if is_grayscale:
        shard_lengths_ans: npt.NDArray[np.uint32] = shard_counts[0].ravel().astype(np.uint32)
        all_cum_stack: npt.NDArray[np.uint64] = gr_cum
        all_sym_stack: npt.NDArray[np.uint64] = gr_sym
    else:
        shard_lengths_ans: npt.NDArray[np.uint32] = shard_counts.ravel().astype(np.uint32)
        all_cum_stack: npt.NDArray[np.uint64] = np.concatenate((gr_cum, rd_cum, bd_cum))
        all_sym_stack: npt.NDArray[np.uint64] = all_sym_freqs_flat

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
            # Use memoryview for zero-copy slicing when appending to bytearray
            sharded_payload += memoryview(bitstreams_flat[off : off + int(bs_lengths[idx])])

    # Step 4: Encode alpha (Zstd)
    if is_rgba:
        c_alpha = zstd.ZstdCompressor(level=1).compress(res_a.tobytes())
        sharded_payload += np.array([len(c_alpha)], dtype='<u4').tobytes() + c_alpha

    modes_diag = np.zeros((3, n_shards), dtype=np.uint8)
    modes_diag[0] = gr_modes
    if not is_grayscale:
        modes_diag[1] = rd_modes
        modes_diag[2] = bd_modes
    return b"SPX_CORE" + header + bytes(sharded_payload) + metadata_bytes, modes_diag


def unpack_bitstream(compressed_data: Union[bytes, BinaryIO], h: int, w: int, is_rgba: bool, is_grayscale: bool,
                     shard_widths: npt.NDArray[np.uint16], shard_modes: npt.NDArray[np.uint8],
                     flag: int) -> Tuple[npt.NDArray[np.uint8], Optional[npt.NDArray[np.uint8]], Optional[npt.NDArray[np.uint8]], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32], Optional[npt.NDArray[np.uint8]]]:
    """
    Deserializes the SPX bitstream format back into parsed frequency tables and residuals.

    Processing Steps:
    1. Decompacts the Zstd-compressed PDF frequency tables and reconstructs CDF arrays for rANS decoding.
    2. Reads shard counts to determine output allocation per shard.
    3. Parallel rANS decoding via ThreadPoolExecutor (GIL released by Numba workers).
    """
    if isinstance(compressed_data, bytes):
        src_mv = memoryview(compressed_data)
    else:
        src_mv = compressed_data

    def read_bytes(src: Union[memoryview, BinaryIO], n: int, pos: int) -> Tuple[Union[memoryview, bytes], int]:
        if isinstance(src, memoryview):
            if pos + n > len(src):
                raise ValueError(f"Unexpected End of File: Expected {n} bytes at {pos}, got {len(src)-pos}")
            return src[pos : pos + n], pos + n
        else:
            data = src.read(n)
            if len(data) < n:
                raise ValueError(f"Unexpected End of Stream: Expected {n} bytes, got {len(data)}")
            return data, 0 # Pos is not used for streams

    def read_block_meta_stream(src: Union[bytes, BinaryIO], pos: int) -> Tuple[bytes, int]:
        b_raw, pos = read_bytes(src, 4, pos)
        b_len = int.from_bytes(b_raw, 'little')
        payload, pos = read_bytes(src, b_len, pos)
        return payload, pos

    dctx = zstd.ZstdDecompressor()

    if flag & FLAG_BITPLANE:
        if is_grayscale:
            profile = PROFILE_RGB
            gray_res = decompress_bitplane_gray_sharded(
                compressed_data, h, w,
                profile.shard_map, profile.noise_shard_id
            ).flatten()
            return gray_res, None, None, None, None, None, None, None
        else:
            # RGB bitplane is decoded upstream in decompress.py via decompress_bitplane_rgb_sharded
            # before unpack_bitstream is ever called — this branch is unreachable in normal flow.
            raise NotImplementedError("RGB bitplane should not reach unpack_bitstream.")

    profile = PROFILE_RGB
    n_shards = profile.total_shards

    p: int = 0
    # 1. Read compacted PDF frequencies
    pdf_raw_bytes, p = read_block_meta_stream(src_mv, p)
    pdf_raw: npt.NDArray[np.uint8] = np.frombuffer(dctx.decompress(pdf_raw_bytes), dtype=np.uint8)
    
    if is_grayscale:
        widths_flat: npt.NDArray[np.uint16] = shard_widths[0, :n_shards].flatten()
        modes_flat: npt.NDArray[np.uint8] = shard_modes[0, :n_shards].flatten()
        all_sym_freqs_flat: npt.NDArray[np.uint64] = expand_pdf_tables(pdf_raw, widths_flat, modes_flat)
        all_sym_freqs = np.zeros((3, n_shards, 256), dtype=np.uint64)
        all_sym_freqs[0] = all_sym_freqs_flat.reshape((n_shards, 256))
        all_cum_freqs: npt.NDArray[np.uint64] = np.zeros((3, n_shards, 257), dtype=np.uint64)
        all_cum_freqs[0, :, 1:] = np.cumsum(all_sym_freqs[0], axis=1)
    else:
        widths_flat: npt.NDArray[np.uint16] = shard_widths[:, :n_shards].flatten()
        modes_flat: npt.NDArray[np.uint8] = shard_modes[:, :n_shards].flatten()
        all_sym_freqs_flat: npt.NDArray[np.uint64] = expand_pdf_tables(pdf_raw, widths_flat, modes_flat)
        all_sym_freqs = all_sym_freqs_flat.reshape((3, n_shards, 256))
        all_cum_freqs: npt.NDArray[np.uint64] = np.zeros((3, n_shards, 257), dtype=np.uint64)
        all_cum_freqs[:, :, 1:] = np.cumsum(all_sym_freqs, axis=2)

    # 2. Read Shard Counts
    sc_raw_bytes, p = read_block_meta_stream(src_mv, p)
    sc_block: npt.NDArray[np.uint32] = np.frombuffer(sc_raw_bytes, dtype='<u4')
    
    shard_counts: npt.NDArray[np.uint32] = np.zeros((3, n_shards), dtype=np.uint32)
    if is_grayscale:
        shard_counts[0] = sc_block.copy().reshape((n_shards,))
    else:
        shard_counts = sc_block.copy().reshape((3, n_shards))

    # 3. Parallel rANS Decoding
    all_lookups: npt.NDArray[np.uint8] = build_all_lookups(all_cum_freqs)

    # Flatten stacks to maximize multi-core thread saturation
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
    if h == 0 or w == 0:
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8), np.zeros(n_shards, dtype=np.uint32), np.zeros(n_shards, dtype=np.uint32), np.zeros(n_shards, dtype=np.uint32), shard_counts, None

    total_res = int(np.sum(counts_stack))
    all_res_flat: npt.NDArray[np.uint8] = np.empty(total_res, dtype=np.uint8)
    
    out_offsets = np.zeros(num_targets, dtype=np.uint32)
    if num_targets > 0:
        out_offsets[1:] = np.cumsum(counts_stack[:-1], dtype=np.uint32)

    # Parallel rANS Decoder: dispatches shard decodes to thread workers concurrently.
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for idx in range(num_targets):
            # 1. Read shard header: 4×uint64 states (as 2×uint32 each) + 1×uint32 length = 36 bytes
            h_raw, p = read_bytes(src_mv, 36, p)
            chunk_meta: npt.NDArray[np.uint32] = np.frombuffer(h_raw, dtype='<u4')
            
            s0 = np.uint64(chunk_meta[0]) | (np.uint64(chunk_meta[1]) << 32)
            s1 = np.uint64(chunk_meta[2]) | (np.uint64(chunk_meta[3]) << 32)
            s2 = np.uint64(chunk_meta[4]) | (np.uint64(chunk_meta[5]) << 32)
            s3 = np.uint64(chunk_meta[6]) | (np.uint64(chunk_meta[7]) << 32)
            b_len = int(chunk_meta[8])
            
            # 2. Read Shard Payload
            b_stream_raw, p = read_bytes(src_mv, b_len, p)
            b_stream = np.frombuffer(b_stream_raw, dtype=np.uint8)
            
            target_cnt = int(counts_stack[idx])
            if target_cnt > 0:
                out_start = int(out_offsets[idx])
                out_view = all_res_flat[out_start : out_start + target_cnt]
                
                # 3. Dispatch to Numba-JIT worker (Releases GIL)
                futures.append(executor.submit(
                    rans_decode_4way_core,
                    s0, s1, s2, s3,
                    b_stream,
                    all_cum_stack[idx],
                    all_sym_stack[idx],
                    all_lookups_stack[idx],
                    out_view
                ))

        # 4. Synchronization Barrier: Wait for all shards to complete
        for future in concurrent.futures.as_completed(futures):
            future.result() # Propagates any internal errors

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
        rd_offs = (out_offsets[n_shards:2*n_shards].astype(np.int64) - gr_len).astype(np.uint32)
        bd_offs = (out_offsets[2*n_shards:].astype(np.int64) - (gr_len + rd_len)).astype(np.uint32)

    res_a_flat: Optional[npt.NDArray[np.uint8]] = None
    if is_rgba:
        a_raw_bytes, p = read_block_meta_stream(src_mv, p)
        res_a_flat = np.frombuffer(dctx.decompress(a_raw_bytes), dtype=np.uint8)
        
    return res_gr_flat, res_rd_flat, res_bd_flat, gr_offs, rd_offs, bd_offs, shard_counts, res_a_flat
