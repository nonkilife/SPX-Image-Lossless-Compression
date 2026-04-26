"""
SPX v1.0.0-stable [4-Pillar Orchestrator]
Module: compress
Role: Compressor Orchestrator.

Architecture:
1. High-throughput lossless image encoder utilizing the 4-pillar modular core.
2. Coder Selection Gate: Dynamically chooses between Sharded rANS (low entropy) 
   and Bitplane rANS (high entropy) based on Pass 1 statistics.
3. Thread-Local Workspace: Optimized memory reuse for concurrent encoding tasks.

Technical Flow:
```mermaid
graph TD
    Input[Input RGB/RGBA] --> Fused[Fused RCT/Pass 1: Decorrelation & Profiling]
    Fused --> CoderSel{H < Threshold & HitRate > Threshold?}
    CoderSel -->|No| Standard[Standard rANS: Sharded Entropy Coding]
    CoderSel -->|Yes| Bitplane[Bitplane rANS: Layered Contextual Coding]
    Standard & Bitplane --> Pack[Final Serialization: SPX Bitstream]
```
"""

__version__ = "1.0.0"

__all__ = [
    'compress_spx',
    'set_parallel_threads',
    'extract_png_metadata',
    'check_grayscale_robust',
]

import numpy as np
import numpy.typing as npt
import logging
import os
import time
from typing import Optional
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import zstandard as zstd
from .common import (
    FLAG_RGBA, FLAG_GRAYSCALE, FLAG_COLOR_GSUB, FLAG_BITPLANE,
    BITPLANE_H_THRESHOLD, BITPLANE_HIT_RATE_THRESHOLD, BITPLANE_P90_THRESHOLD
)
from .sharding import (
    PROFILE_RGB, Pass1Result, execute_sharding_stateless,
    SpxResult, extract_srb_metadata
)
from . import spx_rans as _rs
from .codec import pack_bitstream
from .rans_bitplane import compress_bitplane_gray_sharded, compress_bitplane_rgb_sharded
from . import env

# --- Startup: Validate Dependencies ---
env.verify_environment()

# --- Logging: Core Framework ---
logger: logging.Logger = logging.getLogger("spx.compress")

# [v1.0.0] Module-level Thread-Local for compressor object reuse

def set_parallel_threads(n: int):
    """ 
    Configures the number of CPU threads used by the Rayon pool in the Rust backend.
    Setting this affects all subsequent native calls (Pass 1, Pass 2, and rANS).
    """
    logger.info(f"Parallel threads set to {n} (Rayon-controlled).")

def extract_png_metadata(filepath: str) -> bytes:
    """ 
    Extracts all non-critical PNG metadata chunks (text, time, phys, etc.)
    to preserve original image properties after SPX reconstruction.
    """
    if filepath is None or not os.path.exists(filepath): return b''
    try:
        with open(filepath, 'rb') as f:
            if f.read(8) != b'\x89PNG\r\n\x1a\n': return b''
            out = bytearray()
            while True:
                len_bytes = f.read(4)
                if not len_bytes or len(len_bytes) < 4: break
                length = int.from_bytes(len_bytes, 'big')
                ctype = f.read(4)
                # Skip critical chunks that are reconstructed from the pixel data
                if ctype in [b'IHDR', b'IDAT', b'IEND']:
                    f.seek(length + 4, 1)
                elif length > 10 * 1024 * 1024: # Safety limit: skip massive chunks
                    f.seek(length + 4, 1)
                else:
                    data = f.read(length); crc = f.read(4)
                    out.extend(len_bytes + ctype + data + crc)
                if ctype == b'IEND': break
        return bytes(out)
    except Exception: return b''

def check_grayscale_robust(arr: npt.NDArray[np.uint8], img_mode: Optional[str] = None) -> bool:
    """ 
    [v1.0.0] Optimized Grayscale Detection. 
    Performs a fast sample-check before a full-array scan to minimize latency for RGB images.
    """
    if arr.ndim == 2 or img_mode in ('L', 'LA'): return True
    h, w = arr.shape[0], arr.shape[1]
    if h > 10 and w > 10:
        sample_pts = [(0,0), (h-1, 0), (0, w-1), (h-1, w-1), (h//2, w//2)]
        for r, c in sample_pts:
            px = arr[r, c]
            if not (px[0] == px[1] == px[2]): return False
    return bool(np.array_equal(arr[...,0], arr[...,1]) and np.array_equal(arr[...,1], arr[...,2]))

def _evaluate_coder_selection(shard_counts: npt.NDArray[np.uint32], 
                               shard_widths: npt.NDArray[np.uint16],
                               stats: npt.NDArray[np.uint32],
                               hits_total_p1: npt.NDArray[np.uint32]) -> bool:
    """ 
    Heuristic Gate for Entropy Coder Selection.
    Decides if the image should use Sharded rANS (better for clean images)
    or Bitplane rANS (better for noisy/high-entropy images).
    
    1. Shannon Entropy (H): Average entropy across channels. 
       If H > 3.2, standard context sharding begins to struggle.
    2. Hit-Rate: Fraction of residuals that are exactly zero. 
       High hit-rates favor standard context sharding.
    3. P90 Width: The spread of residuals. 
       If residuals are too wide (>112), bitplane context derivation is too expensive.
    """
    active_mask = shard_counts > 0
    p90_width = float(np.percentile(shard_widths[active_mask], 90)) if active_mask.any() else 256.0
    global_hist = stats.sum(axis=1)
    h_vals = []
    for c_idx in range(3):
        total = float(global_hist[c_idx].sum())
        if total > 0:
            probs = global_hist[c_idx].astype(np.float64) / total
            mask = probs > 0
            h_vals.append(-float(np.sum(probs[mask] * np.log2(probs[mask]))))
    H = float(np.mean(h_vals)) if h_vals else 8.0
    total_hits = float(hits_total_p1.sum())
    total_px = float(shard_counts.sum())
    hit_rate = total_hits / total_px if total_px > 0 else 0.0
    return H < BITPLANE_H_THRESHOLD and hit_rate > BITPLANE_HIT_RATE_THRESHOLD and p90_width < BITPLANE_P90_THRESHOLD

def compress_spx(img_path: Optional[str], output_path: Optional[str] = None,
                  preloaded_arr: Optional[npt.NDArray[np.uint8]] = None,
                  use_bitplane: Optional[bool] = None) -> SpxResult:
    """ 
    Main SPX Compression Entry Point (v1.0.0 Stable). 
    
    Workflow:
    1. Load/Normalize Input: Ensure RGB/RGBA 8-bit format.
    2. Fused Pass 1 (Rust): Perform RCT, MED Prediction, and Shard Profiling in a single pass.
    3. Coder Selection: Evaluate if the Bitplane engine should be triggered.
    4. Pass 2 (Gather): Direct residuals into shard payloads (if not in bitplane mode).
    5. Bitstream Pack: Final rANS entropy coding and container serialization.
    """
    t0: float = time.time()
    try:
        arr: npt.NDArray[np.uint8]
        actual_mode: Optional[str] = None
        if preloaded_arr is not None:
            arr = preloaded_arr
            h, w = arr.shape[0], arr.shape[1]
            c = arr.shape[2] if arr.ndim == 3 else 1
            is_rgba = (c == 4)
        else:
            img: Image.Image = Image.open(img_path)
            img.load()
            actual_mode = img.mode
            target_mode: str = 'RGBA' if img.mode == 'RGBA' else 'RGB'
            img_rgb = img.convert(target_mode)
            arr = np.array(img_rgb)
            h, w = arr.shape[0], arr.shape[1]
            c = arr.shape[2] if arr.ndim == 3 else 1
            is_rgba = (c == 4)

        orig_size: int = os.path.getsize(img_path) if img_path else arr.nbytes
        is_grayscale: bool = check_grayscale_robust(arr, actual_mode)
        
        profile = PROFILE_RGB
        n_shards = profile.total_shards
        s_lut, i_lut, d_lut = profile.spatial_lut, profile.intensity_lut, profile.dispatch_lut
        a_map: npt.NDArray[np.uint8] = arr[:, :, 3].copy() if is_rgba else np.empty((0, 0), dtype=np.uint8)

        if is_grayscale:
            # Grayscale: Single-channel profiling path (Rust)
            gray_raw = arr if arr.ndim == 2 else np.ascontiguousarray(arr[:, :, 0])
            p1_raw = _rs.p1_gray(h, w, gray_raw, a_map, is_rgba, n_shards, s_lut, i_lut, d_lut)
            _gray_ch_hists = np.zeros((3, 256), dtype=np.uint32)
            _gray_ch_hists[0] = p1_raw[2][0].sum(axis=0)
            p1 = Pass1Result(p1_raw[0], np.empty((0,0), np.uint8), np.empty((0,0), np.uint8), a_map, p1_raw[1], p1_raw[2], p1_raw[3], p1_raw[4], p1_raw[5], p1_raw[6], _gray_ch_hists, p1_raw[7], p1_raw[8])
        else:
            # RGB: Fused RCT + 3-channel profiling path (Rust)
            p1_raw = _rs.p1_rgb(h, w, arr, a_map, is_rgba, n_shards, s_lut, i_lut, d_lut)
            p1 = Pass1Result(p1_raw[0], p1_raw[1], p1_raw[2], a_map, p1_raw[4], p1_raw[5], p1_raw[6], p1_raw[7], p1_raw[8], p1_raw[9], p1_raw[3], p1_raw[10], p1_raw[11])

        shard_widths = extract_srb_metadata(p1.shard_stats)
        if use_bitplane is None:
            # Grayscale is always bitplane: single channel means no chroma sharding benefit.
            use_bitplane = True if is_grayscale else _evaluate_coder_selection(p1.shard_counts, shard_widths, p1.shard_stats, p1.hits)

        sbuffer = None
        if not use_bitplane:
            sbuffer = execute_sharding_stateless(h, w, p1, profile, is_grayscale)
        
        metadata_bytes: bytes = extract_png_metadata(img_path)
        metadata_len: int = len(metadata_bytes)
        
        flag: int = int(is_rgba)
        if is_grayscale: flag |= FLAG_GRAYSCALE
        if not use_bitplane: flag |= FLAG_COLOR_GSUB # G-sub is always on for sharded RGB

        selected_mode = "GRAY" if is_grayscale else "RGB"
        final_payload: bytes = b""
        # Populated only in the sharded path; stays zero for bitplane images (no per-shard PDF modes).
        modes_diag: npt.NDArray[np.uint8] = np.zeros((3, n_shards), dtype=np.uint8)

        if use_bitplane:
            bit_payload = bytearray(compress_bitplane_gray_sharded(h, w, p1.gr_ch_p, p1.res_cached[0], profile) if is_grayscale else compress_bitplane_rgb_sharded(h, w, p1.gr_ch_p, p1.res_cached[0], p1.res_cached[1], p1.res_cached[2], profile))
            if is_rgba:
                c_alpha = zstd.ZstdCompressor(level=1).compress(p1.res_cached[3].tobytes())
                bit_payload.extend(np.array([len(c_alpha)], dtype='<u4').tobytes()); bit_payload.extend(c_alpha)
            flag |= FLAG_BITPLANE
            if not is_grayscale: flag |= FLAG_COLOR_GSUB # Bitplane-RGB always uses RCT
            final_payload = b"SPX_CORE" + np.array([h, w, metadata_len, flag], dtype='<u4').tobytes() + bytes(bit_payload) + metadata_bytes
        else:
            final_payload, modes_diag = pack_bitstream(h, w, is_rgba, is_grayscale, True, sbuffer, metadata_bytes, profile)

        if output_path:
            with open(output_path, 'wb') as f_out: f_out.write(final_payload)
        return SpxResult(enc_time=time.time()-t0, h=h, w=w, is_rgba=is_rgba, comp_size=len(final_payload), orig_size=orig_size, hits=p1.hits, res_sums=p1.sums, shard_counts=p1.shard_counts, shard_stats=p1.shard_stats, shard_widths=shard_widths, shard_modes=modes_diag, channel_hists=p1.channel_hists, channels=(p1.gr_ch_p[1:-1, 1:-1], p1.rd_ch_p[1:-1, 1:-1] if not is_grayscale else p1.rd_ch_p, p1.bd_ch_p[1:-1, 1:-1] if not is_grayscale else p1.bd_ch_p, a_map), payload=final_payload, mode=selected_mode)
    except Exception as e:
        logger.error(f"Compression Failure: {e}"); raise
