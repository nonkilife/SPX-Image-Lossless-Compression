"""
SPX v8.3.2-stable [4-Pillar Orchestrator]
Module: spx_compress
Role: Compressor Orchestrator.
Description: High-throughput lossless image encoder utilizing the 4-pillar modular core.
Architecture: Dispatcher layer connecting RGB input to the BICC/rANS pipeline via Fused RCT/Pass 1 kernels.

Technical Flowchart:
```mermaid
graph TD
    Ar[Input RGB/RGBA] --> Fused[Fused RCT/Pass 1: Decorrelation & Profiling]
    Fused --> CodecSel{p90_width < Threshold?}
    CodecSel -->|No| Standard[Standard rANS: 8 Modes: 0/3/4-9]
    CodecSel -->|Yes| Bitplane[Bitplane rANS: 2688 Contexts]
    Standard & Bitplane --> Pack[Codec: Pack Bitstream]
    Pack --> Out[SPX Payload]
```
"""

__version__ = "8.3.2"

import numba
import numpy as np
import numpy.typing as npt
import logging
import os, time
from typing import Optional, Tuple
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import zstandard as zstd
import threading
from .common import (
    FLAG_RGBA, FLAG_SIMPLE, FLAG_RAW, FLAG_PASSTHROUGH, FLAG_GRAYSCALE, FLAG_COLOR_GSUB, FLAG_BITPLANE,
    BITPLANE_H_THRESHOLD, BITPLANE_HIT_RATE_THRESHOLD, BITPLANE_P90_THRESHOLD,
    ENABLE_DIAGNOSTICS
)
from .sharding import (
    PROFILE_RGB, execute_sharding, fused_rct_p1_rgb, fused_rct_p1_gray,
    SpxResult, normalize_shard_stats, calculate_channel_stats, extract_srb_metadata
)
from .transform import (
    predict_2d_residuals
)
from .codec import pack_bitstream
from .rans_bitplane import compress_bitplane_gray_sharded, compress_bitplane_rgb_sharded
from . import env

# [v8.3.2] Internal Math Helpers
@numba.njit(cache=True, inline='always')
def _calculate_alpha_metrics(res_a: npt.NDArray[np.uint8]) -> Tuple[np.uint32, np.uint64]:
    """ Calculates hits and abs_sums for Alpha channel diagnostic parity. """
    h, w = res_a.shape
    hits = np.uint32(0)
    abs_sum = np.uint64(0)
    for i in range(h):
        for j in range(w):
            val = res_a[i, j]
            if val == 128:
                hits += 1
            abs_sum += np.uint64(abs(int(val) - 128))
    return hits, abs_sum

def _evaluate_coder_selection(shard_counts: npt.NDArray[np.uint32], 
                              shard_widths: npt.NDArray[np.uint16],
                              normalized_stats: npt.NDArray[np.uint32],
                              hits_total_p1: npt.NDArray[np.uint32]) -> bool:
    """
    [v8.3.2] Encapsulated Coder Selection Gate.
    Determines if Bitplane rANS should be used based on entropy and hit rate.
    """
    active_mask = shard_counts > 0
    p90_width = float(np.percentile(shard_widths[active_mask], 90)) if active_mask.any() else 256.0

    global_hist = normalized_stats.sum(axis=1)  # (3, 256) - collapse shards
    h_vals = []
    for c_idx in range(3):
        total = float(global_hist[c_idx].sum())
        if total > 0:
            probs = global_hist[c_idx].astype(np.float64) / total
            mask = probs > 0
            h_vals.append(-float(np.sum(probs[mask] * np.log2(probs[mask]))))
    H = float(np.mean(h_vals)) if h_vals else 8.0

    total_hits = float(hits_total_p1[0]) + float(hits_total_p1[1]) + float(hits_total_p1[2])
    total_px = float(shard_counts[0].sum() + shard_counts[1].sum() + shard_counts[2].sum())
    hit_rate = total_hits / total_px if total_px > 0 else 0.0

    use_bitplane = bool(
        H < BITPLANE_H_THRESHOLD
        and hit_rate > BITPLANE_HIT_RATE_THRESHOLD
        and p90_width < BITPLANE_P90_THRESHOLD
    )
    logger.debug(
        f"Coder auto-select: H={H:.3f} hit_rate={hit_rate:.3f} p90={p90_width:.1f} "
        f"-> {'bitplane' if use_bitplane else 'standard'}"
    )
    return use_bitplane

# --- Startup: Validate Dependencies ---
env.verify_environment()

# --- Logging: Core Framework ---
logger: logging.Logger = logging.getLogger("spx.compress")


# [v8.3.2] Module-level Thread-Local for compressor object reuse
thread_local_comp: threading.local = threading.local()


def clear_spx_workspaces():
    """ [v5.2.3] Forces release of large Thread-Local memory buffers to prevent OOM in long-lived workers (FastAPI/Celery). """
    if hasattr(thread_local_comp, 'row_shard_hists'):
        del thread_local_comp.row_shard_hists
    if hasattr(thread_local_comp, 'comp'):
        del thread_local_comp.comp
    import gc
    gc.collect()

# =============================================================================
# --- 1. Global Utilities ---
# =============================================================================

def zstandard_compress(data: bytes) -> bytes:
    """ 
    Thread-safe entropy coding utility with compressor reuse. 
    """
    if not hasattr(thread_local_comp, 'comp'):
        thread_local_comp.comp = zstd.ZstdCompressor(level=1, threads=1)
    return thread_local_comp.comp.compress(data)

def set_parallel_threads(n: int):
    """
    [v8.3.2] Configures the number of CPU threads used by the Numba parallel engine.
    """
    numba.set_num_threads(n)
    logger.info(f"Numba Parallel Engine set to {n} thread(s).")

_MAX_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB — OOM DoS guard for non-critical chunks

def extract_png_metadata(filepath: str) -> bytes:
    """
    Extracts raw PNG metadata chunks (excluding IHDR, IDAT, IEND) to allow lossless recreation.
    """
    if filepath is None or not os.path.exists(filepath):
        return b''
    try:
        with open(filepath, 'rb') as f:
            magic: bytes = f.read(8)
            if magic != b'\x89PNG\r\n\x1a\n':
                return b''
            out: bytearray = bytearray()
            while True:
                len_bytes: bytes = f.read(4)
                if not len_bytes or len(len_bytes) < 4:
                    break
                length: int = int.from_bytes(len_bytes, 'big')
                ctype: bytes = f.read(4)
                
                # [v5.1] Memory Protection: Limit non-critical chunk sizes to prevent OOM DoS.
                if ctype in [b'IHDR', b'IDAT', b'IEND']:
                    f.seek(length + 4, 1) # Skip Data + CRC
                else:
                    if length > _MAX_CHUNK_SIZE:
                        logger.warning(f"Chunk {ctype.decode(errors='replace')} exceeds safe limit ({length} bytes). Skipping.")
                        f.seek(length + 4, 1) # Skip oversized chunk
                    else:
                        data: bytes = f.read(length)
                        crc: bytes = f.read(4)
                        out.extend(len_bytes + ctype + data + crc)
                
                if ctype == b'IEND':
                    break
        return bytes(out)
    except Exception as e:
        # Only log error if filepath was actually provided and exists
        if filepath and os.path.exists(filepath):
            logger.error(f"Failed to extract PNG metadata from {filepath}: {e}")
        return b''

# =============================================================================
# --- 2. Diagnostic Helpers ---
# =============================================================================


def check_grayscale_robust(arr: npt.NDArray[np.uint8], img_mode: Optional[str] = None) -> bool:
    """ 
    [v8.3.2] Hybrid Grayscale Detection: Metadata -> Sampling -> Full Verify.
    Designed for 100% Correctness with High Performance.
    """
    # Phase 0: Dimension Check
    if arr.ndim == 2:
        return True

    # Phase 1: Metadata Truth
    if img_mode in ('L', 'LA'):
        return True
    
    # Phase 2: Heuristic Sampling (Fast-Negative)
    h, w = arr.shape[0], arr.shape[1]
    if h > 10 and w > 10:
        # Check 5 strategic points: corners and center
        sample_pts = [(0,0), (h-1, 0), (0, w-1), (h-1, w-1), (h//2, w//2)]
        for r, c in sample_pts:
            px = arr[r, c]
            if not (px[0] == px[1] == px[2]):
                return False
                
    # Phase 3: Comprehensive Verification (Last Resort)
    # R vs G, G vs B for bit-perfect guarantee
    return bool(np.array_equal(arr[...,0], arr[...,1]) and np.array_equal(arr[...,1], arr[...,2]))

def dump_shard_stats(shard_stats: npt.NDArray[np.uint32], img_name: str):
    """ [v4.6.0-SIM] Captures 87 histograms for offline entropy sharding research. """
    if not os.environ.get("SPX_DUMP_SHARDS"):
        return
        
    out_dir = "_debug_shards"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # img_name might be a full path, get basename
    base_name = os.path.basename(img_name)
    np.savez_compressed(os.path.join(out_dir, f"{base_name}.npz"), stats=shard_stats)

def compress_spx(img_path: Optional[str], output_path: Optional[str] = None,
                  preloaded_arr: Optional[npt.NDArray[np.uint8]] = None,
                  force_mode: Optional[int] = None,
                  use_bitplane: Optional[bool] = None) -> SpxResult:
    """ 
    Main SPX Compression Entry Point (v8.3.2 Stable).
    
    Orchestrates the massive parallel encoding pipeline:
    1. G-Sub RCT Transform -> 2. Pass 1 Shard Histograms -> 3. Median-Normalization
    4. Pass 2 Residual Vectorization -> 5. 4-Way SIMD rANS Entropy Coding -> 6. Container Packaging.
    
    NOTE: Utilizes extremely aggressive Numba internal parallelism (`prange`).
    """
    t0: float = time.time()
    
    try:
        arr: npt.NDArray[np.uint8]
        # [v2.16] Optimization: Use pre-loaded array if available to avoid redundant I/O
        actual_mode: Optional[str] = None
        if preloaded_arr is not None:
            arr = preloaded_arr
            h, w = arr.shape[0], arr.shape[1]
            c = arr.shape[2] if arr.ndim == 3 else 1
            is_rgba: bool = (c == 4)
        else:
            # Standard Path
            img: Image.Image = Image.open(img_path)
            img.load()
            actual_mode = img.mode
            target_mode: str = 'RGBA' if img.mode == 'RGBA' else 'RGB'
            img_rgb: Image.Image = img.convert(target_mode)
            arr = np.array(img_rgb)
            h, w = arr.shape[0], arr.shape[1]
            c = arr.shape[2] if arr.ndim == 3 else 1
            is_rgba: bool = (c == 4)

        # [v4.0.8.2] Secure Original Size Tracking
        orig_size: int = 0
        try:
            orig_size = os.path.getsize(img_path) if img_path else arr.nbytes
        except (TypeError, OSError):
            orig_size = arr.nbytes

        # [v6.6] Optimized Grayscale Detection (Metadata aware)
        is_grayscale: bool = check_grayscale_robust(arr, actual_mode)
        
        # 3. Fused RCT + Shard Pass 1 — eliminates separate extract_channels scan and np.pad copies.
        pixels: int = h * w
        profile = PROFILE_RGB
        n_shards: int = profile.total_shards
        s_lut, i_lut, d_lut = profile.spatial_lut, profile.intensity_lut, profile.dispatch_lut

        # [v6.6] Unified RGB Sharding
        logger.debug(f"Entropy Profile: RGB")

        # G-sub is applied inside fused kernel; this flag only marks the bitstream header.
        use_gsub = True

        a_map: npt.NDArray[np.uint8] = arr[:, :, 3].copy() if is_rgba else np.empty((0, 0), dtype=np.uint8)

        if is_grayscale:
            gray_raw = arr if arr.ndim == 2 else arr[:, :, 0]
            (gr_map_p, _rd_p, _bd_p, _ch_hists,
             shard_counts, shard_stats, shard_offsets_p1, row_global_offsets,
             (hits_total_p1, sums_total_p1), res_cached) = fused_rct_p1_gray(
                h, w, gray_raw, a_map, is_rgba, n_shards, s_lut, i_lut, d_lut)
            rd_map_p = np.empty((0, 0), dtype=np.uint8)
            bd_map_p = np.empty((0, 0), dtype=np.uint8)
            channel_hists = np.zeros((3, 256), dtype=np.uint32)
        else:
            (gr_map_p, rd_map_p, bd_map_p, channel_hists,
             shard_counts, shard_stats, shard_offsets_p1, row_global_offsets,
             (hits_total_p1, sums_total_p1), res_cached) = fused_rct_p1_rgb(
                h, w, arr, a_map, is_rgba, n_shards, s_lut, i_lut, d_lut)

        ctx_map = np.empty((0, 0, 0), dtype=np.uint8)
        p1_cached = (shard_counts, shard_stats, shard_offsets_p1, row_global_offsets,
                     (hits_total_p1, sums_total_p1), res_cached, ctx_map)

        # Calculate Global Modes for Noise Shard Prediction
        modes: npt.NDArray[np.uint8] = np.zeros(3, dtype=np.uint8)
        for c_idx in range(3):
            mode_val = calculate_channel_stats(channel_hists[c_idx])
            modes[c_idx] = np.uint8(mode_val)
        
        # [v8.0] Static Residual Normalization: Transform centered Pass 1 stats to normalized ZigZag stats
        normalized_stats: npt.NDArray[np.uint32] = normalize_shard_stats(shard_stats)
        
        # [v4.7.2-STABLE] Metadata Extraction (NOW USING NORMALIZED RANGES)
        shard_widths: npt.NDArray[np.uint16]
        shard_widths = extract_srb_metadata(normalized_stats)

        # [v8.3.2] Per-Image Coder Selection: auto-detect bitplane vs standard rANS.
        # Gate: H < 3.2 AND hit_rate > 0.30 AND p90 < 112 (Tightened for stability)
        # H = mean Shannon entropy across 3 channels; hit_rate = zero-residual fraction.
        # The caller can override by passing use_bitplane=True/False explicitly.
        if use_bitplane is None:
            if is_grayscale:
                use_bitplane = True
            else:
                use_bitplane = _evaluate_coder_selection(shard_counts, shard_widths, normalized_stats, hits_total_p1)

        # [v8.3.2] Standard Path: Execute Full Sharding Hub
        if not use_bitplane:
            sbuffer = execute_sharding(h, w, gr_map_p, rd_map_p, bd_map_p, a_map, is_rgba, is_grayscale, profile, p1_cached=p1_cached)
            
            if is_rgba:
                hits_total_p1 = np.concatenate((hits_total_p1, np.array([np.uint32(sbuffer.a_metrics[0])], dtype=np.uint32)))
                sums_total_p1 = np.concatenate((sums_total_p1, np.array([np.uint64(sbuffer.a_metrics[1])], dtype=np.uint64)))

        
        metadata_bytes: bytes = extract_png_metadata(img_path)
        metadata_len: int = len(metadata_bytes)
        
        # 1. Determine initial mode via gating
        is_too_small: bool = (pixels < 1024)

        # Evaluate fallbacks only if necessary
        size_simple: float = 1e18
        simple_payload: bytes = b""
        raw_bytes: bytes = b""
        # [v4.9.1 Fix] Optimization: Only check SIMPLE for tiny icons (<64k pixels)
        if force_mode == 1 or pixels < 65536:
            if not raw_bytes:
                raw_bytes = gr_map_p[1:-1, 1:-1].tobytes() if is_grayscale else arr.tobytes()
            simple_payload = zstandard_compress(raw_bytes)
            size_simple = 8 + 16 + len(simple_payload) + metadata_len

        size_raw: int = 8 + 16 + h * w * c + metadata_len
        # Passthrough is only an option if the physical file exists
        size_pass: float = 8 + 16 + orig_size if (img_path and os.path.exists(img_path)) else 1e18

        force_simple: bool = (force_mode == 1) or is_too_small
        selected_mode: str = "RGB"
        if force_simple:
            selected_mode = "SIMPLE" if size_simple < size_raw else "RAW"

        flag: int = int(is_rgba)
        if selected_mode == "SIMPLE": flag |= FLAG_SIMPLE
        elif selected_mode == "RAW": flag |= FLAG_RAW
        
        # [v6.0] Sharded mode
        is_sharded_path = (selected_mode == "RGB")
        if is_sharded_path:
            if is_grayscale: 
                flag |= FLAG_GRAYSCALE
                # [v6.6] Ensure mode string reflects grayscale detection
                selected_mode = "GRAY"
            if use_gsub: flag |= FLAG_COLOR_GSUB

        header_base: bytes = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
        
        final_payload: bytes = b""
        modes_diag: npt.NDArray[np.uint8] = np.zeros((3, n_shards), dtype=np.uint8)
        if selected_mode == "SIMPLE":
            final_payload = b"SPX_CORE" + header_base + simple_payload + metadata_bytes
        elif selected_mode == "RAW":
            final_payload = b"SPX_CORE" + header_base + raw_bytes + metadata_bytes
        elif use_bitplane and is_grayscale:
            # [v7.3] Shard-Conditioned Bitplane Grayscale Path
            bit_payload = bytearray(compress_bitplane_gray_sharded(
                h, w,
                gr_map_p, res_cached[0],
                profile
            ))
            if is_rgba:
                res_a = res_cached[3]
                # [v8.3.1] Diagnostic Consistency
                hits_total_p1 = np.concatenate((hits_total_p1, np.array([np.uint32(res_cached[4][0])], dtype=np.uint32)))
                sums_total_p1 = np.concatenate((sums_total_p1, np.array([np.uint64(res_cached[4][1])], dtype=np.uint64)))
                c_alpha = zstd.ZstdCompressor(level=1).compress(res_a.tobytes())
                bit_payload.extend(np.array([len(c_alpha)], dtype='<u4').tobytes())
                bit_payload.extend(c_alpha)
            flag |= FLAG_BITPLANE
            header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
            final_payload = b"SPX_CORE" + header_base + bytes(bit_payload) + metadata_bytes
        elif use_bitplane and selected_mode == "RGB":
            # [v7.3] Shard-Conditioned Bitplane RGB Path
            bit_payload = bytearray(compress_bitplane_rgb_sharded(
                h, w,
                gr_map_p, res_cached[0], res_cached[1], res_cached[2],
                profile
            ))
            if is_rgba:
                res_a = res_cached[3]
                # [v8.3.1] Diagnostic Consistency
                hits_total_p1 = np.concatenate((hits_total_p1, np.array([np.uint32(res_cached[4][0])], dtype=np.uint32)))
                sums_total_p1 = np.concatenate((sums_total_p1, np.array([np.uint64(res_cached[4][1])], dtype=np.uint64)))
                c_alpha = zstd.ZstdCompressor(level=1).compress(res_a.tobytes())
                bit_payload.extend(np.array([len(c_alpha)], dtype='<u4').tobytes())
                bit_payload.extend(c_alpha)
            flag |= FLAG_BITPLANE
            header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
            final_payload = b"SPX_CORE" + header_base + bytes(bit_payload) + metadata_bytes
        else:
            final_payload, modes_diag = pack_bitstream(
                h, w, is_rgba, is_grayscale, use_gsub,
                sbuffer, metadata_bytes, profile
            )
            
        # [v5.2.2] Emergency Downgrade Protection (Lazy Evaluation)
        # If SHARDED/GRAY expands (common for high-entropy noise), calculate SIMPLE fallback if skipped earlier.
        if selected_mode in ("RGB", "GRAY") and len(final_payload) > size_raw:
            if not simple_payload:
                if not raw_bytes:
                    raw_bytes = gr_map_p[1:-1, 1:-1].tobytes() if is_grayscale else arr.tobytes()
                simple_payload = zstandard_compress(raw_bytes)
                size_simple = 8 + 16 + len(simple_payload) + metadata_len
            
        if len(final_payload) > size_simple or len(final_payload) > size_pass or len(final_payload) > size_raw:
            # Select the absolute best among available fallbacks
            min_size = min(size_simple, size_pass, size_raw)
            if size_pass == min_size:
                with open(img_path, 'rb') as f_orig: original_bytes: bytes = f_orig.read()
                flag = (flag & (FLAG_RGBA | FLAG_GRAYSCALE)) | FLAG_PASSTHROUGH
                header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
                final_payload = b"SPX_CORE" + header_base + original_bytes
                selected_mode = "PASSTHROUGH"
            elif size_simple == min_size:
                flag = (flag & (FLAG_RGBA | FLAG_GRAYSCALE)) | FLAG_SIMPLE
                header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
                final_payload = b"SPX_CORE" + header_base + simple_payload + metadata_bytes
                selected_mode = "SIMPLE"
            else:
                # Fallback to RAW (Uncompressed raw pixels)
                flag = (flag & (FLAG_RGBA | FLAG_GRAYSCALE)) | FLAG_RAW
                header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
                if not raw_bytes:
                    raw_bytes = gr_map_p[1:-1, 1:-1].tobytes() if is_grayscale else arr.tobytes()
                final_payload = b"SPX_CORE" + header_base + raw_bytes + metadata_bytes
                selected_mode = "RAW"

        if output_path:
            with open(output_path, 'wb') as f_out: f_out.write(final_payload)

        res_modes: npt.NDArray[np.uint8] = modes_diag

        # Release resources
        clear_spx_workspaces()

        return SpxResult(enc_time=time.time()-t0, h=h, w=w, is_rgba=is_rgba, comp_size=len(final_payload),
                          orig_size=orig_size, hits=hits_total_p1, res_sums=sums_total_p1,
                          shard_counts=shard_counts,
                          shard_ptrs=None,
                          shard_stats=shard_stats,
                          shard_widths=shard_widths,
                          shard_modes=res_modes,
                          channel_hists=channel_hists,
                          channel_modes=modes,
                          channels=(gr_map_p[1:-1, 1:-1], rd_map_p[1:-1, 1:-1] if not is_grayscale else rd_map_p, bd_map_p[1:-1, 1:-1] if not is_grayscale else bd_map_p, a_map),
                          payload=final_payload,
                          mode=selected_mode)
    except Exception as e:
        logger.error(f"Compression Failure: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        raise
