"""
SPX v8.3.2 [Stable Parallel Architecture]
Module: rans
Role: Pillar 4 - Entropy Engine (Unified).

Description: 
Thin Python shim over the spx_rans Rust native backend. This module provides 
the high-level interface for the Asymmetric Numeral Systems (ANS) entropy 
engine, specifically the range-based variant (rANS).

Architecture & Engineering Rationale:
1. Interleaved rANS: The encoder uses a 4-way interleaved architecture. This 
   allows the decoder to process four rANS states in parallel, maximizing 
   Instruction-Level Parallelism (ILP) and overcoming the serial bottleneck 
   inherent in standard rANS.
2. Renormalization (L_LOWER): The lower bound for the rANS state (2^31). 
   This ensures that the state never drops below a threshold where it would 
   lose precision or overflow during symbol addition.
3. PDF Compaction: To save header space, custom frequency tables are compacted 
   using a 12-bit serialization format and Zstd compression. This is only 
   used if an Empirical Template (Mode 4-33) doesn't provide sufficient fit.

All heavy computation (PDF building, rANS encode/decode, serialization)
is handled by the spx_rans Rust extension.
"""

__version__ = "8.3.2"

__all__ = [
    'L_LOWER',
    'build_pdf_tables_from_shards',
    'rans_encode_shards_parallel',
    'rans_decode_shards_parallel',
    'rans_decode_4way_core',
    'pack_shard_payloads',
    'build_all_lookups',
    'compact_pdf_tables',
    'expand_pdf_tables',
]

import os
import numpy as np
import numpy.typing as npt
from typing import Tuple

import spx_rans as _rs
from .common import get_empirical_templates

# ---------------------------------------------------------------------------
# Scalar constants (used by codec.py)
# ---------------------------------------------------------------------------

# L_LOWER = 1 << 31. This is the normalization boundary for the 64-bit rANS state.
L_LOWER: np.uint64 = np.uint64(2147483648)

# ---------------------------------------------------------------------------
# _MAGIC_LUT  — fetched from Rust so there is exactly one source of truth.
# This LUT contains precomputed reciprocals for fast frequency division 
# during rANS state transitions.
# ---------------------------------------------------------------------------

_MAGIC_LUT: npt.NDArray[np.uint64] = _rs.get_magic_lut()

# ---------------------------------------------------------------------------
# Public API — delegates to Rust, preserving the exact call signatures used
# by codec.py so no changes are needed there.
# ---------------------------------------------------------------------------

def _get_templates():
    """Returns (templates_array, disable_flag) for passing to Rust functions."""
    tpl = get_empirical_templates()           # (30, 256) uint64
    disable = os.environ.get("SPX_DISABLE_TEMPLATES") == "1"
    return tpl, disable


def build_pdf_tables_from_shards(
    data_flat: npt.NDArray[np.uint8],
    shard_offsets: npt.NDArray[np.uint32],
    shard_lengths: npt.NDArray[np.uint32],
    shard_widths: npt.NDArray[np.uint16],
) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64], npt.NDArray[np.uint8]]:
    """
    Builds cumulative + symbol frequency tables for all shards in parallel.
    Uses Rayon in Rust to scan shard data and determine the best-fit 
    statistical model (Empirical Template or Custom PDF).
    """
    tpl, disable = _get_templates()
    return _rs.build_pdf_tables_from_shards(
        data_flat.ravel().astype(np.uint8),
        shard_offsets.ravel().astype(np.uint32),
        shard_lengths.ravel().astype(np.uint32),
        shard_widths.ravel().astype(np.uint16),
        tpl.astype(np.uint64),
        bool(disable),
    )


def rans_encode_shards_parallel(
    shard_data_flat: npt.NDArray[np.uint8],
    shard_offsets: npt.NDArray[np.uint32],
    shard_lengths: npt.NDArray[np.uint32],
    all_cum_freqs: npt.NDArray[np.uint64],
    all_sym_freqs: npt.NDArray[np.uint64],
    initial_state: np.uint64,
) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint8], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """
    Parallel 4-way interleaved rANS encoding over all shards.
    Each shard is processed by a separate thread, and each thread interleaves 
    4 rANS states to saturate the CPU's execution units.
    """
    return _rs.rans_encode_shards_parallel(
        np.ascontiguousarray(shard_data_flat, dtype=np.uint8),
        np.ascontiguousarray(shard_offsets, dtype=np.uint32),
        np.ascontiguousarray(shard_lengths, dtype=np.uint32),
        np.ascontiguousarray(all_cum_freqs, dtype=np.uint64),
        np.ascontiguousarray(all_sym_freqs, dtype=np.uint64),
        np.uint64(initial_state),
    )


def rans_decode_shards_parallel(
    payload: npt.NDArray[np.uint8],
    shard_counts: npt.NDArray[np.uint32],
    cum_freqs: npt.NDArray[np.uint64],
    sym_freqs: npt.NDArray[np.uint64],
    lookups: npt.NDArray[np.uint8],
) -> Tuple[npt.NDArray[np.uint8], int]:
    """
    Batch rANS decode: all shards decoded in parallel via Rayon. 
    Returns (residuals_flat, bytes_consumed).
    This is the highest-throughput decoding path.
    """
    return _rs.rans_decode_shards_parallel(
        np.ascontiguousarray(payload, dtype=np.uint8),
        np.ascontiguousarray(shard_counts, dtype=np.uint32),
        np.ascontiguousarray(cum_freqs, dtype=np.uint64),
        np.ascontiguousarray(sym_freqs, dtype=np.uint64),
        np.ascontiguousarray(lookups, dtype=np.uint8),
    )


def rans_decode_4way_core(
    st0: np.uint64, st1: np.uint64, st2: np.uint64, st3: np.uint64,
    bitstream: npt.NDArray[np.uint8],
    cum_freqs: npt.NDArray[np.uint64],
    symbol_freqs: npt.NDArray[np.uint64],
    slot_lookup: npt.NDArray[np.uint8],
    out: npt.NDArray[np.uint8],
) -> None:
    """
    4-way interleaved rANS decode (in-place into `out`).
    Lower-level primitive for decoding a single shard's bitstream.
    """
    _rs.rans_decode_4way_core(
        np.uint64(st0), np.uint64(st1), np.uint64(st2), np.uint64(st3),
        np.ascontiguousarray(bitstream, dtype=np.uint8),
        np.ascontiguousarray(cum_freqs, dtype=np.uint64),
        np.ascontiguousarray(symbol_freqs, dtype=np.uint64),
        np.ascontiguousarray(slot_lookup, dtype=np.uint8),
        out,  # passed directly — must already be C-contiguous
    )


def pack_shard_payloads(
    final_states: npt.NDArray[np.uint64],
    bs_lengths: npt.NDArray[np.uint32],
    bs_offsets: npt.NDArray[np.uint32],
    bitstreams_flat: npt.NDArray[np.uint8],
    out: npt.NDArray[np.uint8],
    shard_write_offsets: npt.NDArray[np.uint32],
) -> None:
    """
    Serializes all shard payloads into a pre-allocated buffer (in-place).
    Ensures that each shard can be independently accessed by its offset.
    """
    _rs.pack_shard_payloads(
        np.ascontiguousarray(final_states, dtype=np.uint64),
        np.ascontiguousarray(bs_lengths, dtype=np.uint32),
        np.ascontiguousarray(bs_offsets, dtype=np.uint32),
        np.ascontiguousarray(bitstreams_flat, dtype=np.uint8),
        out,
        np.ascontiguousarray(shard_write_offsets, dtype=np.uint32),
    )


def build_all_lookups(
    all_cum_freqs: npt.NDArray[np.uint64],
) -> npt.NDArray[np.uint8]:
    """
    Batch-precomputes slot-lookup tables for all (channel, shard) pairs.
    Slot-lookups allow O(1) symbol derivation during rANS decoding.
    """
    return _rs.build_all_lookups(
        np.ascontiguousarray(all_cum_freqs, dtype=np.uint64)
    )


def compact_pdf_tables(
    all_sym_freqs: npt.NDArray[np.uint64],
    shard_widths: npt.NDArray[np.uint16],
    shard_modes: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    """Serializes custom frequency tables into a compact bytes buffer."""
    return _rs.compact_pdf_tables(
        np.ascontiguousarray(all_sym_freqs, dtype=np.uint64),
        np.ascontiguousarray(shard_widths, dtype=np.uint16),
        np.ascontiguousarray(shard_modes, dtype=np.uint8),
    )


def expand_pdf_tables(
    compacted_data: npt.NDArray[np.uint8],
    shard_widths: npt.NDArray[np.uint16],
    shard_modes: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint64]:
    """Reconstructs 256-symbol probability arrays from a compacted byte stream."""
    tpl, _ = _get_templates()
    return _rs.expand_pdf_tables(
        np.ascontiguousarray(compacted_data, dtype=np.uint8),
        np.ascontiguousarray(shard_widths, dtype=np.uint16),
        np.ascontiguousarray(shard_modes, dtype=np.uint8),
        tpl.astype(np.uint64),
    )
