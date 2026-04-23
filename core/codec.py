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
    Meta[Header: H, W, Flags] --> SRB[SRB Block: Widths, Modes]
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
import threading
from typing import Tuple, List, Optional, Union, BinaryIO, NamedTuple
from .common import (
    FLAG_RGBA, FLAG_GRAYSCALE, FLAG_COLOR_GSUB, FLAG_BITPLANE,
    FLAG_SIMPLE, FLAG_RAW, FLAG_PASSTHROUGH
)
from .sharding import (
    PROFILE_RGB, ShardProfile, ShardBuffer,
    normalize_shard_stats, extract_srb_metadata
)
from .rans import (
    rans_encode_shards_parallel,
    build_pdf_tables_from_shards, L_LOWER,
    compact_pdf_tables,
    rans_decode_4way_core, build_all_lookups,
    expand_pdf_tables
)
 
# [v8.2.1] Thread-Local for compressor reuse
thread_local_codec = threading.local()

def get_zstd_comp(level: int = 3) -> zstd.ZstdCompressor:
    """Provides a thread-safe cached ZstdCompressor."""
    key = f"comp_{level}"
    if not hasattr(thread_local_codec, key):
        setattr(thread_local_codec, key, zstd.ZstdCompressor(level=level))
    return getattr(thread_local_codec, key)

class SpxUnpackResult(NamedTuple):
    """Structured container for SPX decompression artifacts."""
    h: int
    w: int
    flag: int
    res_gr: npt.NDArray[np.uint8]
    res_rd: npt.NDArray[np.uint8]
    res_bd: npt.NDArray[np.uint8]
    gr_offs: npt.NDArray[np.uint32]
    rd_offs: npt.NDArray[np.uint32]
    bd_offs: npt.NDArray[np.uint32]
    shard_counts: npt.NDArray[np.uint32]
    res_a: Optional[npt.NDArray[np.uint8]]
    payload: bytes
    metadata: bytes

def pack_bitstream(h: int, w: int, is_rgba: bool, is_grayscale: bool, use_gsub: bool,
                     sbuffer: ShardBuffer,
                     metadata_bytes: bytes,
                     profile: ShardProfile = PROFILE_RGB) -> Tuple[bytes, npt.NDArray[np.uint8]]:
    """
    Serializes compressed data into the final SPX file block.
    [v8.2.1] Optimized ShardBuffer Serialization.
    """
    flag: int = FLAG_RGBA if is_rgba else 0
    if is_grayscale: flag |= FLAG_GRAYSCALE
    if use_gsub: flag |= FLAG_COLOR_GSUB

    n_shards = profile.total_shards
    n_channels = 1 if is_grayscale else 3
    
    shard_counts = sbuffer.counts
    # [v8.2.0] Automatic Width Extraction from ShardBuffer Stats
    normalized_stats = normalize_shard_stats(sbuffer.stats)
    shard_widths = extract_srb_metadata(normalized_stats)

    # 1. SRB Block: Widths and Modes (Unified)
    header_widths = (shard_widths[:n_channels, :n_shards] % 256).astype(np.uint8).tobytes()
    
    # 2. Build PDF tables and extract modes (Unified Channel Stacking)
    all_shards = []
    
    # [v8.2.1] Reconstruct shard views directly from channel payloads to avoid redundant slicing of res_flat
    src_channels = [sbuffer.gr_payload] if is_grayscale else [sbuffer.gr_payload, sbuffer.rd_payload, sbuffer.bd_payload]
    for c_idx, c_data in enumerate(src_channels):
        c_counts = shard_counts[c_idx]
        shard_offs = np.zeros(n_shards + 1, dtype=np.uint32)
        shard_offs[1:] = np.cumsum(c_counts)
        for s in range(n_shards):
            all_shards.append(c_data[shard_offs[s]:shard_offs[s+1]])

    c_cums, c_syms, c_modes = build_pdf_tables_from_shards(all_shards, shard_widths[:n_channels].ravel())
    
    # 3. Compact and Compress PDF Block
    pdf_compact = compact_pdf_tables(c_syms, shard_widths[:n_channels, :n_shards].ravel(), c_modes)
    c_pdf = get_zstd_comp(level=3).compress(pdf_compact.tobytes())
    
    # 4. Assemble Payload Structure
    payload_parts = []
    payload_parts.append(np.uint32(len(c_pdf)).tobytes())
    payload_parts.append(c_pdf)
    
    sc_out = shard_counts[:n_channels].tobytes()
    payload_parts.append(np.uint32(len(sc_out)).tobytes())
    payload_parts.append(sc_out)

    # 5. Parallel rANS Encoding
    # Concatenate payloads for rANS processing at the latest possible stage
    if is_grayscale:
        res_flat = sbuffer.gr_payload
    else:
        res_flat = np.concatenate([sbuffer.gr_payload, sbuffer.rd_payload, sbuffer.bd_payload])

    shard_lengths_ans = shard_counts[:n_channels].ravel().astype(np.uint32)
    shard_offsets_ans = np.zeros(len(shard_lengths_ans), dtype=np.uint32)
    if len(shard_lengths_ans) > 0:
        shard_offsets_ans[1:] = np.cumsum(shard_lengths_ans[:-1])
        
    final_states, bitstreams_flat, bs_offsets, bs_lengths = rans_encode_shards_parallel(
        res_flat, shard_offsets_ans, shard_lengths_ans, c_cums, c_syms, L_LOWER
    )
    
    for idx in range(len(shard_lengths_ans)):
        payload_parts.append(final_states[idx].astype('<u8').tobytes())
        payload_parts.append(np.uint32(bs_lengths[idx]).tobytes())
        if bs_lengths[idx] > 0:
            off = int(bs_offsets[idx])
            payload_parts.append(bitstreams_flat[off : off + int(bs_lengths[idx])])

    # 6. Optional Alpha Layer
    if is_rgba:
        c_alpha = get_zstd_comp(level=1).compress(sbuffer.a_payload.tobytes())
        payload_parts.append(np.uint32(len(c_alpha)).tobytes())
        payload_parts.append(c_alpha)

    # Final Serialization
    header_base = np.array([h, w, len(metadata_bytes), flag], dtype='<u4').tobytes()
    modes_diag = np.zeros((3, n_shards), dtype=np.uint8)
    modes_diag[:n_channels] = c_modes.reshape((n_channels, n_shards))
    
    full_blob = b"".join([b"SPX_CORE", header_base, header_widths, c_modes.tobytes(), *payload_parts, metadata_bytes])
    return full_blob, modes_diag


def unpack_bitstream(compressed_data: Union[bytes, BinaryIO], profile: ShardProfile = PROFILE_RGB) -> SpxUnpackResult:
    """
    Deserializes the SPX bitstream format.
    [v8.2.1] Unified Transport Layer (Protocol Symmetry).
    """
    if not isinstance(compressed_data, bytes) and not compressed_data.seekable():
        compressed_data = compressed_data.read()

    if isinstance(compressed_data, bytes):
        src_mv = memoryview(compressed_data)
        src_len = len(src_mv)
    else:
        src_mv = compressed_data
        src_mv.seek(0, 2)
        src_len = src_mv.tell()
        src_mv.seek(0)

    def read_bytes(n: int, pos: int) -> Tuple[Union[memoryview, bytes], int]:
        if isinstance(src_mv, memoryview): return src_mv[pos : pos + n], pos + n
        src_mv.seek(pos); return src_mv.read(n), pos + n

    def read_block_meta(pos: int) -> Tuple[bytes, int]:
        b_raw, pos = read_bytes(4, pos)
        b_len = int.from_bytes(b_raw, 'little')
        payload, pos = read_bytes(b_len, pos)
        return bytes(payload), pos

    p = 0
    magic, p = read_bytes(8, p)
    if magic != b"SPX_CORE": raise ValueError("Invalid SPX Magic String")
    
    h_base_raw, p = read_bytes(16, p)
    h, w, m_len, flag = np.frombuffer(h_base_raw, dtype='<u4')
    is_grayscale, is_rgba = bool(flag & FLAG_GRAYSCALE), bool(flag & FLAG_RGBA)
    n_shards = profile.total_shards
    n_channels = 1 if is_grayscale else 3

    meta_start = src_len - m_len
    metadata, _ = read_bytes(m_len, meta_start)
    
    # Mode-specific Bypass (Simple/Bitplane)
    if flag & (FLAG_SIMPLE | FLAG_RAW | FLAG_PASSTHROUGH):
        payload, _ = read_bytes(meta_start - p, p)
        return SpxUnpackResult(h, w, flag, np.empty(0, np.uint8), np.empty(0, np.uint8), np.empty(0, np.uint8), 
                               np.zeros(n_shards, np.uint32), np.zeros(n_shards, np.uint32), np.zeros(n_shards, np.uint32), 
                               np.zeros((3, n_shards), np.uint32), None, bytes(payload), bytes(metadata))

    if flag & FLAG_BITPLANE:
        payload_for_bp, _ = read_bytes(src_len - m_len - p, p)
        return SpxUnpackResult(h, w, flag, np.empty(0, np.uint8), np.empty(0, np.uint8), np.empty(0, np.uint8), 
                               np.zeros(n_shards, np.uint32), np.zeros(n_shards, np.uint32), np.zeros(n_shards, np.uint32), 
                               np.zeros((3, n_shards), np.uint32), None, bytes(payload_for_bp), bytes(metadata))

    # 1. Shard Metadata (Widths & Modes)
    srb_len = n_channels * n_shards
    widths_raw, p = read_bytes(srb_len, p)
    modes_raw, p = read_bytes(srb_len, p)
    
    r_widths = np.frombuffer(widths_raw, dtype=np.uint8).reshape((n_channels, n_shards))
    shard_widths = np.where(r_widths == 0, np.uint16(256), r_widths.astype(np.uint16))
    shard_modes = np.frombuffer(modes_raw, dtype=np.uint8).reshape((n_channels, n_shards))

    # 2. PDF Block Reconstruction
    pdf_c_bytes, p = read_block_meta(p)
    pdf_raw = zstd.ZstdDecompressor().decompress(pdf_c_bytes)
    all_sym_freqs_flat = expand_pdf_tables(np.frombuffer(pdf_raw, np.uint8), shard_widths.ravel(), shard_modes.ravel())
    
    all_sym_freqs = np.zeros((3, n_shards, 256), dtype=np.uint64)
    all_sym_freqs[:n_channels] = all_sym_freqs_flat.reshape((n_channels, n_shards, 256))
    all_cum_freqs = np.zeros((3, n_shards, 257), dtype=np.uint64)
    all_cum_freqs[:n_channels, :, 1:] = np.cumsum(all_sym_freqs[:n_channels], axis=2)

    # 3. Shard Counts
    sc_raw, p = read_block_meta(p)
    shard_counts = np.zeros((3, n_shards), dtype=np.uint32)
    shard_counts[:n_channels] = np.frombuffer(sc_raw, dtype='<u4').reshape((n_channels, n_shards))

    # 4. Parallel rANS Decoding Dispatch
    all_lookups = build_all_lookups(all_cum_freqs)
    counts_stack = shard_counts[:n_channels].ravel()
    cum_stack = all_cum_freqs[:n_channels].reshape((-1, 257))
    sym_stack = all_sym_freqs[:n_channels].reshape((-1, 256))
    lookups_stack = all_lookups[:n_channels].reshape((-1, 4096))
    
    total_res = int(np.sum(counts_stack))
    all_res_flat = np.empty(total_res, dtype=np.uint8)
    out_offsets = np.zeros(len(counts_stack), dtype=np.uint32)
    if len(counts_stack) > 0: out_offsets[1:] = np.cumsum(counts_stack[:-1])

    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for idx in range(len(counts_stack)):
            h_raw, p = read_bytes(36, p)
            meta = np.frombuffer(h_raw, dtype='<u4')
            s0 = np.uint64(meta[0]) | (np.uint64(meta[1]) << 32)
            s1 = np.uint64(meta[2]) | (np.uint64(meta[3]) << 32)
            s2 = np.uint64(meta[4]) | (np.uint64(meta[5]) << 32)
            s3 = np.uint64(meta[6]) | (np.uint64(meta[7]) << 32)
            b_len, b_stream_raw = int(meta[8]), None
            b_stream_raw, p = read_bytes(b_len, p)
            if counts_stack[idx] > 0:
                out_view = all_res_flat[out_offsets[idx] : out_offsets[idx] + counts_stack[idx]]
                futures.append(executor.submit(rans_decode_4way_core, s0, s1, s2, s3, 
                                            np.frombuffer(b_stream_raw, np.uint8), 
                                            cum_stack[idx], sym_stack[idx], lookups_stack[idx], out_view))
        for f in concurrent.futures.as_completed(futures): f.result()

    # 5. Recombine Channels (Unified n_channels logic)
    ch_lens = [int(np.sum(shard_counts[i])) for i in range(3)]
    res_split = [all_res_flat[sum(ch_lens[:i]) : sum(ch_lens[:i+1])] for i in range(3)]
    
    all_offs = out_offsets.reshape((n_channels, n_shards))
    ch_offs = [ (all_offs[i] - sum(ch_lens[:i])).astype(np.uint32) if i < n_channels else np.zeros(n_shards, np.uint32) for i in range(3)]

    res_a = None
    if is_rgba:
        a_c, p = read_block_meta(p)
        res_a = np.frombuffer(zstd.ZstdDecompressor().decompress(a_c), dtype=np.uint8)
        
    return SpxUnpackResult(h, w, flag, res_split[0], res_split[1], res_split[2], 
                           ch_offs[0], ch_offs[1], ch_offs[2], shard_counts, res_a, b"", bytes(metadata))
