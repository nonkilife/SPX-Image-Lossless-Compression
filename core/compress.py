"""
ZPNG-CSDE v6.2 [Flexible-Shard Architecture]
Module: zpng_compress
Role: Compressor Orchestrator.
Description: High-throughput lossless image encoder utilizing the 4-pillar modular core.
Architecture: Dispatcher layer connecting RGB input to the BICC/rANS pipeline via Flexible Sharding Hub.

Technical Flowchart:
```mermaid
graph TD
    Ar[Input RGB/RGBA] --> GSUB[G-sub RCT: Decorrelate Channels]
    GSUB --> USH[Universal Sharding Hub: 42 Contexts]
    
    USH --> Pass1[Pass 1: Histograms & Bias Estimation]
    Pass1 --> BICC[BICC: Shard-Level Centroid Shifting]
    
    BICC --> Pass2[Pass 2: Optimized Residual Mapping]
    Pass2 --> rANS[rANS: 4-way Parallel Entropy Coding]
    rANS --> Out[ZStandard Container Payload]
```
"""

import numba
import numpy as np
import numpy.typing as npt
import logging
import os, time
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import zstandard as zstd
from numba import njit, prange, uint8, uint32, uint64
import concurrent.futures
import threading, sys
from .rans import (
    rans_encode_shards_parallel, 
    build_pdf_tables_from_shards, L_LOWER,
    compact_pdf_tables
)

from .common import (
    ZpngResult, FLAG_RGBA, FLAG_SIMPLE, FLAG_RAW, FLAG_PASSTHROUGH, FLAG_GRAYSCALE, FLAG_COLOR_GSUB, FLAG_BITPLANE,
    TOTAL_SHARDS, apply_median_to_stats,
    calculate_channel_stats, PROFILE_RGB,
    extract_srb_metadata, to_zigzag, from_zigzag, predict_med_standard,
    PREDICTOR_LUT, BITPLANE_WIDTH_THRESHOLD, BITPLANE_MIN_PIXELS
)
from .transform import (
    extract_channels, predict_2d_residuals,
    calculate_aad_estimate
)
from .shard_rgb import (
    predict_pass_1, predict_pass_2
)
from .shard_gray import (
    predict_pass_1_gray, predict_pass_2_gray
)
from .codec import pack_bitstream_v5, pack_bitplane_bitstream_v5
from .rans_bitplane_sharded import compress_bitplane_gray_sharded, compress_bitplane_rgb_sharded
from . import env

# --- Startup: Validate Dependencies ---
env.verify_environment()

# --- Logging: Core Framework ---
logger: logging.Logger = logging.getLogger("zpng.compress")


# [v2.25] Module-level Thread-Local for compressor object reuse
thread_local_comp: threading.local = threading.local()

def get_thread_local_workspace(h: int) -> npt.NDArray[np.uint32]:
    """ [v5.2] Manages a persistent workspace for row_shard_hists to avoid 200MB+ daily allocation. """
    target_h = max(h, 2160) # Target 2K baseline, or current h
    if not hasattr(thread_local_comp, 'row_shard_hists'):
        thread_local_comp.row_shard_hists = np.zeros((target_h, 3, TOTAL_SHARDS, 256), dtype=np.uint32)
    elif thread_local_comp.row_shard_hists.shape[0] < h:
        # Resize if current image is larger than our buffer
        thread_local_comp.row_shard_hists = np.zeros((h, 3, TOTAL_SHARDS, 256), dtype=np.uint32)
    
    workspace = thread_local_comp.row_shard_hists[:h]
    workspace.fill(0) # Faster than re-allocation + zeroing by OS
    return workspace

def clear_zpng_workspaces():
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
    [v5.2.4] Configures the number of CPU threads used by the Numba parallel engine.
    """
    numba.set_num_threads(n)
    logger.info(f"Numba Parallel Engine set to {n} thread(s).")

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
                MAX_CHUNK_SIZE = 10 * 1024 * 1024 # 10MB
                
                if ctype in [b'IHDR', b'IDAT', b'IEND']:
                    f.seek(length + 4, 1) # Skip Data + CRC
                else:
                    if length > MAX_CHUNK_SIZE:
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
# --- 3. Diagnostic Helpers ---
# =============================================================================

def analyze_shard_ranges(shard_stats: npt.NDArray[np.uint32], verbose: bool = False):
    """ [Research] Analyzes the actual residual spread in each shard. """
    print("\n--- [Research] Shard Residual Range Analysis ---")
    total_widths = 0.0
    shard_count = 0
    max_observed_width = 0
    
    # Pre-map zigzag to signed
    vals = np.zeros(256, dtype=np.int16)
    for i in range(256):
        z8 = np.uint8(i)
        mask = -np.int16(z8 & 1)
        vals[i] = np.int16(z8 >> 1) ^ mask

    for c in range(3):
        chan_name = ["Grn", "RD", "BD"][c]
        for k in range(TOTAL_SHARDS):
            hist = shard_stats[c, k]
            count = np.sum(hist)
            if count == 0: continue
            
            indices = np.where(hist > 0)[0]
            if len(indices) == 0: continue
            
            shard_vals = vals[indices]
            min_v, max_v = np.min(shard_vals), np.max(shard_vals)
            width = int(max_v - min_v + 1)
            
            total_widths += width
            shard_count += 1
            max_observed_width = max(max_observed_width, width)
            
    if shard_count > 0:
        avg_width = total_widths / shard_count
        print(f"有效分片數: {shard_count}")
        print(f"平均殘差寬度: {avg_width:.2f} (理論值: 256)")
        print(f"最大觀測寬度: {max_observed_width}")
        print(f"理論空間節省: {(1.0 - avg_width/256.0)*100:.2f}%")
    print("-----------------------------------------------\n")

def check_grayscale_robust(arr: npt.NDArray[np.uint8], img_mode: Optional[str] = None) -> bool:
    """ 
    [v6.6] Hybrid Grayscale Detection: Metadata -> Sampling -> Full Verify.
    Designed for 100% Correctness with High Performance.
    """
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
    if not os.environ.get("ZPNG_DUMP_SHARDS"):
        return
        
    out_dir = "_debug_shards"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # img_name might be a full path, get basename
    base_name = os.path.basename(img_name)
    np.savez_compressed(os.path.join(out_dir, f"{base_name}.npz"), stats=shard_stats)

def compress_csde(img_path: Optional[str], output_path: Optional[str] = None,
                  preloaded_arr: Optional[npt.NDArray[np.uint8]] = None,
                  force_mode: Optional[int] = None,
                  use_bitplane: Optional[bool] = None) -> ZpngResult:
    """ 
    Main ZPNG-CSDE Compression Entry Point (V6.6 Stable).
    
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
            h, w, c = arr.shape
            is_rgba: bool = (c == 4)
        else:
            # Standard Path
            img: Image.Image = Image.open(img_path)
            img.load()
            actual_mode = img.mode
            target_mode: str = 'RGBA' if img.mode == 'RGBA' else 'RGB'
            img_rgb: Image.Image = img.convert(target_mode)
            arr = np.array(img_rgb)
            h, w, c = arr.shape
            is_rgba: bool = (c == 4)

        # [v4.0.8.2] Secure Original Size Tracking
        orig_size: int = 0
        try:
            orig_size = os.path.getsize(img_path) if img_path else arr.nbytes
        except (TypeError, OSError):
            orig_size = arr.nbytes

        # [v6.6] Optimized Grayscale Detection (Metadata aware)
        is_grayscale: bool = check_grayscale_robust(arr, actual_mode)
        
        # 3. Channel Extraction & Statistical Prep
        # USH utilizes the GradMean (AAD) estimate for reporting/diagnostics.
        pixels: int = h * w
        gr_map, rd_map, bd_map, a_map, channel_hists = extract_channels(arr)
        
        # [v5.5] Calculate AAD for diagnostic reporting only
        aad_val: float = calculate_aad_estimate(gr_map)
        
        # [v6.6] Unified RGB Sharding
        from .common import ShardProfile, sync_luts_if_needed
        profile: ShardProfile = PROFILE_RGB
        
        # [v6.6 Defensive] Ensure global Context LUTs match requested profile
        sync_luts_if_needed(profile.v_boundaries_gr, profile.intensity_segments)
        
        n_shards: int = profile.total_shards
        
        # [Diagnostic] Mode Signaling
        logger.debug(f"Entropy Profile: RGB | AAD: {aad_val:.4f}")
        
        shard_counts: npt.NDArray[np.uint32] = np.zeros((3, n_shards), dtype=np.uint32)
        shard_stats: npt.NDArray[np.uint32] = np.zeros((3, n_shards, 256), dtype=np.uint32)
        
        use_gsub = True # Standardized
 
        # Calculate Global Modes for Noise Shard Prediction
        modes: npt.NDArray[np.uint8] = np.zeros(3, dtype=np.uint8)
        for c_idx in range(3):
            _, _, mode_val = calculate_channel_stats(channel_hists[c_idx])
            modes[c_idx] = np.uint8(mode_val)
        
        if is_grayscale:
            shard_counts, shard_stats, shard_offsets_p1, row_global_offsets, shard_medians, \
            (hits_total_p1, sums_total_p1) = \
                predict_pass_1_gray(h, w, gr_map,
                                    profile.shard_map, profile.v_boundaries_gr,
                                    profile.intensity_segments, profile.noise_shard_id,
                                    PREDICTOR_LUT)
        else:
            shard_counts, shard_stats, shard_offsets_p1, row_global_offsets, shard_medians, \
            (hits_total_p1, sums_total_p1) = \
                predict_pass_1(h, w, gr_map, rd_map, bd_map, False,
                               profile.shard_map, profile.v_boundaries_gr,
                               profile.intensity_segments, profile.noise_shard_id)
        

        
        # [v6.5] Median Normalization Alignment: Transform centered Pass 1 stats to normalized ZigZag stats
        biased_stats: npt.NDArray[np.uint32] = apply_median_to_stats(shard_stats, shard_medians)
        
        # [v4.7.2-STABLE] Metadata Extraction (NOW USING NORMALIZED RANGES)
        shard_widths: npt.NDArray[np.uint16]
        _, shard_widths = extract_srb_metadata(biased_stats)

        # [v7.5] Per-Image Coder Selection: auto-detect bitplane vs standard rANS.
        # Grayscale always uses bitplane — the shard-conditioned bitplane coder was
        # specifically tuned for grayscale and consistently wins there.
        # For RGB, use the 90th-percentile ZigZag residual width over active shards.
        # p90 captures tail behaviour: bitplane needs the *entire* distribution to
        # be narrow, not just the average.  Natural images have a few wide high-energy
        # boundary shards that inflate the tail even when the median is low — mean
        # and median are blind to this, p90 is not.  Empirical: Tecnick p90 ≤ 95,
        # DIV2K p90 ≥ 70.5; threshold 85 gives 99% classification accuracy.
        # A minimum pixel gate (BITPLANE_MIN_PIXELS) guards against fixed table
        # overhead dominating on small images (Kodak, small CLIC).
        # The caller can override by passing use_bitplane=True/False explicitly.
        if use_bitplane is None:
            if is_grayscale:
                use_bitplane = True
            else:
                active_mask = shard_counts > 0
                if active_mask.any():
                    p90_width = float(np.percentile(shard_widths[active_mask], 90))
                else:
                    p90_width = 256.0
                pixel_count = h * w
                use_bitplane = bool(
                    p90_width <= BITPLANE_WIDTH_THRESHOLD
                    and pixel_count >= BITPLANE_MIN_PIXELS
                )
                logger.debug(
                    f"Coder auto-select: p90_width={p90_width:.1f} pixels={pixel_count} "
                    f"-> {'bitplane' if use_bitplane else 'standard'}"
                )

        # 5. Zero-Copy rANS Buffer Allocation (v4.9.2)
        total_res_size: int = int(shard_counts.sum())
        all_shards_flat: npt.NDArray[np.uint8] = np.empty(total_res_size, dtype=np.uint8)
        
        # Create zero-copy views for each channel
        gr_size: int = int(shard_counts[0].sum())
        rd_size: int = int(shard_counts[1].sum())
        shard_gr = all_shards_flat[0 : gr_size]
        shard_rd = all_shards_flat[gr_size : gr_size + rd_size]
        shard_bd = all_shards_flat[gr_size + rd_size :]
        
        # 6. Predict & Shard (Pass 2: Direct Write to Unified Buffer)
        res_a: npt.NDArray[np.uint8]
        a_hits: np.uint64
        a_sum: float
        if is_grayscale:
            res_a, (a_hits, a_sum) = \
                predict_pass_2_gray(h, w, gr_map, a_map, is_rgba,
                                    profile.shard_map, profile.v_boundaries_gr, profile.intensity_segments, profile.noise_shard_id,
                                    row_global_offsets, shard_medians, shard_gr,
                                    PREDICTOR_LUT)
        else:
            res_a, (a_hits, a_sum) = \
                predict_pass_2(h, w, gr_map, rd_map, bd_map, a_map, is_rgba, False,
                               profile.shard_map, profile.v_boundaries_gr, profile.intensity_segments, profile.noise_shard_id,
                               row_global_offsets, shard_medians,
                               shard_gr, shard_rd, shard_bd)

        # [v6.5] Phase 1 Delivery: Reporting Median Normalization Statistics
        if os.environ.get("ZPNG_REPORT_MEDIAN"):
            print("\n--- [Phase 1] Median effectiveness report ---")
            for c_idx in range(3):
                chan = ["Grn", "RD", "BD"][c_idx]
                total_samples = shard_counts[c_idx].sum()
                if total_samples > 0:
                    # [v6.5] Raw Zero is now at index 128 due to centered storage
                    raw_zero = shard_stats[c_idx, :, 128].sum()
                    median_sum = 0
                    for s_i in range(n_shards):
                        m_val = shard_medians[c_idx, s_i]
                        median_sum += shard_stats[c_idx, s_i, m_val]
                    
                    raw_occ = (raw_zero / total_samples) * 100
                    med_occ = (median_sum / total_samples) * 100
                    print(f"Channel {chan}: Raw Zero: {raw_occ:.2f}% -> Median-Shifted Zero: {med_occ:.2f}% (Gain: {med_occ-raw_occ:+.2f}%)")
            print("-------------------------------------------\n")

        
        # [Fix] Merge Alpha stats into global diagnostic arrays
        if is_rgba:
            hits_total_p1[3] = np.uint32(a_hits)
            sums_total_p1[3] = np.uint64(a_sum)
        
        metadata_bytes: bytes = extract_png_metadata(img_path)
        metadata_len: int = len(metadata_bytes)
        
        # 1. Determine initial mode via gating (v5.1.2: score reused from pre-pass)
        is_too_small: bool = (pixels < 1024)
        is_too_complex: bool = False # Gating removed

        # Evaluate fallbacks only if necessary
        size_simple: float = 1e18
        simple_payload: bytes = b""
        raw_bytes: bytes = b""
        # [v4.9.1 Fix] Optimization: Only check SIMPLE for tiny icons (<64k) or high-complexity outliers
        # Ensure force_mode == 1 trigger is included in calculation path.
        if force_mode == 1 or is_too_small or is_too_complex or pixels < 65536:
            raw_bytes = arr.tobytes()
            simple_payload = zstandard_compress(raw_bytes)
            size_simple = 8 + 16 + len(simple_payload) + metadata_len

        size_raw: int = 8 + 16 + h * w * c + metadata_len
        # Passthrough is only an option if the physical file exists
        size_pass: float = 8 + 16 + orig_size if (img_path and os.path.exists(img_path)) else 1e18

        force_simple: bool = (force_mode == 1) or is_too_small or is_too_complex
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
        if selected_mode == "SIMPLE":
            final_payload = b"ZPNGCSDE" + header_base + simple_payload + metadata_bytes
        elif selected_mode == "RAW":
            final_payload = b"ZPNGCSDE" + header_base + raw_bytes + metadata_bytes
        elif use_bitplane and is_grayscale:
            # [v7.3] Shard-Conditioned Bitplane Grayscale Path
            resid_2d = predict_2d_residuals(gr_map)
            bit_payload = compress_bitplane_gray_sharded(
                gr_map, resid_2d,
                profile.shard_map, profile.v_boundaries_gr,
                profile.intensity_segments, profile.noise_shard_id
            )
            flag |= FLAG_BITPLANE
            header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
            final_payload = b"ZPNGCSDE" + header_base + bit_payload + metadata_bytes
            selected_mode = "GRAY"
        elif use_bitplane and selected_mode == "RGB":
            # [v7.3] Shard-Conditioned Bitplane RGB Path
            gr_resid = predict_2d_residuals(gr_map)
            rd_resid = predict_2d_residuals(rd_map)
            bd_resid = predict_2d_residuals(bd_map)
            bit_payload = compress_bitplane_rgb_sharded(
                gr_map, rd_map, bd_map,
                gr_resid, rd_resid, bd_resid,
                profile.shard_map, profile.v_boundaries_gr,
                profile.intensity_segments, profile.noise_shard_id
            )
            flag |= FLAG_BITPLANE
            header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
            final_payload = b"ZPNGCSDE" + header_base + bit_payload + metadata_bytes
        else:
            modes_diag: npt.NDArray[np.uint8]
            final_payload, modes_diag = pack_bitstream_v5(
                h, w, is_rgba, is_grayscale, use_gsub,
                shard_counts, shard_offsets_p1, shard_widths, shard_medians,
                all_shards_flat, res_a, metadata_bytes
            )
            
        # [v5.2.2] Emergency Downgrade Protection (Lazy Evaluation)
        # If SHARDED/GRAY expands (common for high-entropy noise), calculate SIMPLE fallback if skipped earlier.
        if selected_mode in ("RGB", "GRAY") and len(final_payload) > size_raw:
            if not simple_payload:
                if not raw_bytes: raw_bytes = arr.tobytes()
                simple_payload = zstandard_compress(raw_bytes)
                size_simple = 8 + 16 + len(simple_payload) + metadata_len
            
        if len(final_payload) > size_simple or len(final_payload) > size_pass or len(final_payload) > size_raw:
            # Select the absolute best among available fallbacks
            min_size = min(size_simple, size_pass, size_raw)
            if size_pass == min_size:
                with open(img_path, 'rb') as f_orig: original_bytes: bytes = f_orig.read()
                flag = (flag & FLAG_RGBA) | FLAG_PASSTHROUGH
                header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
                final_payload = b"ZPNGCSDE" + header_base + original_bytes
            elif size_simple == min_size:
                flag = (flag & FLAG_RGBA) | FLAG_SIMPLE
                header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
                final_payload = b"ZPNGCSDE" + header_base + simple_payload + metadata_bytes
            else:
                # Fallback to RAW (Uncompressed raw pixels)
                flag = (flag & FLAG_RGBA) | FLAG_RAW
                header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
                if not raw_bytes: raw_bytes = arr.tobytes()
                final_payload = b"ZPNGCSDE" + header_base + raw_bytes + metadata_bytes

        if output_path:
            with open(output_path, 'wb') as f_out: f_out.write(final_payload)

        # Re-capture modes for diagnostics even for fallback paths (optional, but good for research)
        res_modes: npt.NDArray[np.uint8] = modes_diag if 'modes_diag' in locals() else np.zeros((3, n_shards), dtype=np.uint8)

        return ZpngResult(enc_time=time.time()-t0, h=h, w=w, is_rgba=is_rgba, comp_size=len(final_payload),
                          orig_size=orig_size, hits=hits_total_p1, res_sums=sums_total_p1,
                          shard_counts=shard_counts,
                          shard_ptrs=None,
                          shard_stats=shard_stats,
                          shard_mins=np.zeros((3, n_shards), dtype=np.uint8),
                          shard_widths=shard_widths,
                          shard_medians=shard_medians,
                          shard_modes=res_modes,
                          channel_hists=channel_hists,
                          channel_modes=modes,
                          channels=(gr_map, rd_map, bd_map, a_map),
                          payload=final_payload,
                          mode=selected_mode,
                          aad=aad_val)
    except Exception as e:
        logger.error(f"Compression Failure: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise

# --- Pytest Snippet for Core Logic Verification ---
"""
def test_compression_parity():
    import numpy as np
    from .compress import compress_csde
    from .decompress import decompress_csde
    
    # Create synthetic test image (G-sub compatible)
    data = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # Compress
    res = compress_csde(None, None, preloaded_arr=data)
    assert res.payload is not None
    assert len(res.payload) > 0
    
    # Decompress
    rec, _ = decompress_csde(res.payload, None)
    
    # MSE Check
    mse = np.mean((data.astype(np.float64) - rec.astype(np.float64))**2)
    assert mse == 0.0, f"Bit-perfect check failed: MSE={mse}"
"""
