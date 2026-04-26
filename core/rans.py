"""
SPX v8.3.2 [Stable Parallel Architecture]
Module: rans
Role: Pillar 4 - Entropy Engine (Unified).
Description: Thin Python shim over the spx_rans Rust native backend.

All heavy computation (PDF building, rANS encode/decode, serialization)
is handled by the spx_rans Rust extension.

See technical/rust_convert.md for migration history and pitfalls.
"""

__version__ = "8.3.2"

import os
import numpy as np
import numpy.typing as npt
from typing import Tuple

import spx_rans as _rs
from .common import get_empirical_templates

# ---------------------------------------------------------------------------
# Scalar constants (used by codec.py)
# ---------------------------------------------------------------------------

L_LOWER: np.uint64 = np.uint64(2147483648)   # 1 << 31

# ---------------------------------------------------------------------------
# _MAGIC_LUT  — fetched from Rust so there is exactly one source of truth.
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
    """Builds cumulative + symbol frequency tables for all shards in parallel."""
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
    """Parallel 4-way interleaved rANS encoding over all shards."""
    return _rs.rans_encode_shards_parallel(
        np.ascontiguousarray(shard_data_flat, dtype=np.uint8),
        np.ascontiguousarray(shard_offsets, dtype=np.uint32),
        np.ascontiguousarray(shard_lengths, dtype=np.uint32),
        np.ascontiguousarray(all_cum_freqs, dtype=np.uint64),
        np.ascontiguousarray(all_sym_freqs, dtype=np.uint64),
        np.uint64(initial_state),
    )


def rans_decode_4way_core(
    st0: np.uint64, st1: np.uint64, st2: np.uint64, st3: np.uint64,
    bitstream: npt.NDArray[np.uint8],
    cum_freqs: npt.NDArray[np.uint64],
    symbol_freqs: npt.NDArray[np.uint64],
    slot_lookup: npt.NDArray[np.uint8],
    out: npt.NDArray[np.uint8],
) -> None:
    """4-way interleaved rANS decode (in-place into `out`)."""
    _rs.rans_decode_4way_core(
        np.uint64(st0), np.uint64(st1), np.uint64(st2), np.uint64(st3),
        np.ascontiguousarray(bitstream, dtype=np.uint8),
        np.ascontiguousarray(cum_freqs, dtype=np.uint64),
        np.ascontiguousarray(symbol_freqs, dtype=np.uint64),
        np.ascontiguousarray(slot_lookup, dtype=np.uint8),
        out,  # passed directly — must already be C-contiguous (it is: np.empty slice)
    )


def pack_shard_payloads(
    final_states: npt.NDArray[np.uint64],
    bs_lengths: npt.NDArray[np.uint32],
    bs_offsets: npt.NDArray[np.uint32],
    bitstreams_flat: npt.NDArray[np.uint8],
    out: npt.NDArray[np.uint8],
    shard_write_offsets: npt.NDArray[np.uint32],
) -> None:
    """Serializes all shard payloads into a pre-allocated buffer (in-place)."""
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
    """Batch-precomputes slot-lookup tables for all (channel, shard) pairs."""
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
