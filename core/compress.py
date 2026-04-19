"""
ZPNG-CSDE v6.2 [Flexible-Shard Architecture]
Module: zpng_compress
Role: Compressor Orchestrator.
Description: High-throughput lossless image encoder utilizing the 4-pillar modular core.
Architecture: Dispatcher layer connecting RGB input to the BICC/rANS pipeline via Flexible Sharding Hub.

Technical Flowchart:
```mermaid
graph TD
    Ar[Input RGB/RGBA] --> GSUB[G-sub RCT: Extract G, RD, BD]
    GSUB --> Pass1[Pass 1: Universal-42 Profiling]
    Pass1 --> CodecSel{p90_width < Threshold?}
    CodecSel -->|No| Standard[Standard rANS: 8 Modes: 0/3/4-9]
    CodecSel -->|Yes| Bitplane[Bitplane rANS: 2688 Contexts]
    Standard & Bitplane --> Pack[Codec: Pack Bitstream]
    Pack --> Out[ZPNG Payload]
```
"""

import numba
import numpy as np
import numpy.typing as npt
import logging
import os, time
from typing import Optional
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import zstandard as zstd
import threading
from .common import (
    ZpngResult, FLAG_RGBA, FLAG_SIMPLE, FLAG_RAW, FLAG_PASSTHROUGH, FLAG_GRAYSCALE, FLAG_COLOR_GSUB, FLAG_BITPLANE,
    calculate_channel_stats, PROFILE_RGB,
    extract_srb_metadata,
    BITPLANE_WIDTH_THRESHOLD, BITPLANE_MIN_PIXELS
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
from .codec import pack_bitstream
from .rans_bitplane import compress_bitplane_gray_sharded, compress_bitplane_rgb_sharded
from . import env

# --- Startup: Validate Dependencies ---
env.verify_environment()

# --- Logging: Core Framework ---
logger: logging.Logger = logging.getLogger("zpng.compress")


# [v2.25] Module-level Thread-Local for compressor object reuse
thread_local_comp: threading.local = threading.local()


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
        sync_luts_if_needed(profile.v_boundaries_gr, profile.intensity_segments, profile.shard_map, profile.noise_shard_id)
        
        n_shards: int = profile.total_shards
        
        # [Diagnostic] Mode Signaling
        logger.debug(f"Entropy Profile: RGB | AAD: {aad_val:.4f}")
        
        shard_counts: npt.NDArray[np.uint32] = np.zeros((3, n_shards), dtype=np.uint32)
        shard_stats: npt.NDArray[np.uint32] = np.zeros((3, n_shards, 256), dtype=np.uint32)
        
        # G-sub is applied inside extract_channels; this flag only marks the bitstream header.
        use_gsub = True
 
        # Calculate Global Modes for Noise Shard Prediction
        modes: npt.NDArray[np.uint8] = np.zeros(3, dtype=np.uint8)
        for c_idx in range(3):
            _, _, mode_val = calculate_channel_stats(channel_hists[c_idx])
            modes[c_idx] = np.uint8(mode_val)
        
        # Pad channel maps with 1-pixel zero border for guard-free Numba kernels
        gr_map_p = np.pad(gr_map, 1, constant_values=0)
        if not is_grayscale:
            rd_map_p = np.pad(rd_map, 1, constant_values=0)
            bd_map_p = np.pad(bd_map, 1, constant_values=0)

        if is_grayscale:
            shard_counts, shard_stats, shard_offsets_p1, row_global_offsets, shard_medians, \
            (hits_total_p1, sums_total_p1) = \
                predict_pass_1_gray(h, w, gr_map_p,
                                    profile.shard_map, profile.noise_shard_id)
        else:
            shard_counts, shard_stats, shard_offsets_p1, row_global_offsets, shard_medians, \
            (hits_total_p1, sums_total_p1) = \
                predict_pass_1(h, w, gr_map_p, rd_map_p, bd_map_p, False,
                               profile.shard_map, profile.noise_shard_id)
        

        
        # [v4.7.2-STABLE] Metadata Extraction
        shard_widths: npt.NDArray[np.uint16]
        shard_widths = extract_srb_metadata(shard_stats)

        # [v7.5] Per-Image Coder Selection: auto-detect bitplane vs standard rANS.
        # Grayscale always uses bitplane - the shard-conditioned bitplane coder was
        # optimized for high-res low-entropy textures.
        
        # p90 capturing tail behavior - bitplane needs the entire distribution
        # to be narrow, not just the median. Natural images have wide high-energy
        # boundary shards that inflate the tail even when the median is low - mean
        # and median are blind to this, p90 is not. Empirical: Tecnick p90 > 95,
        # DIV2K p90 > 70.5; threshold 85 gives 99% classification accuracy.
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

        # 5 & 6. Buffer allocation and Pass 2 are only needed for standard rANS.
        # Bitplane path uses predict_2d_residuals and does not consume shard buffers.
        if not use_bitplane:
            total_res_size: int = int(shard_counts.sum())
            all_shards_flat: npt.NDArray[np.uint8] = np.empty(total_res_size, dtype=np.uint8)
            gr_size: int = int(shard_counts[0].sum())
            rd_size: int = int(shard_counts[1].sum())
            shard_gr = all_shards_flat[0 : gr_size]
            shard_rd = all_shards_flat[gr_size : gr_size + rd_size]
            shard_bd = all_shards_flat[gr_size + rd_size :]

            res_a: npt.NDArray[np.uint8]
            a_hits: np.uint64
            a_sum: float
            if is_grayscale:
                res_a, (a_hits, a_sum) = \
                    predict_pass_2_gray(h, w, gr_map_p, a_map, is_rgba,
                                        profile.shard_map, profile.noise_shard_id,
                                        row_global_offsets, shard_medians, shard_gr)
            else:
                res_a, (a_hits, a_sum) = \
                    predict_pass_2(h, w, gr_map_p, rd_map_p, bd_map_p, a_map, is_rgba, False,
                                   profile.shard_map, profile.noise_shard_id,
                                   row_global_offsets, shard_medians,
                                   shard_gr, shard_rd, shard_bd)

            if is_rgba:
                hits_total_p1[3] = np.uint32(a_hits)
                sums_total_p1[3] = np.uint64(a_sum)

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
            raw_bytes = arr.tobytes()
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
            final_payload = b"ZPNGCSDE" + header_base + simple_payload + metadata_bytes
        elif selected_mode == "RAW":
            final_payload = b"ZPNGCSDE" + header_base + raw_bytes + metadata_bytes
        elif use_bitplane and is_grayscale:
            # [v7.3] Shard-Conditioned Bitplane Grayscale Path
            resid_2d = predict_2d_residuals(gr_map)
            bit_payload = compress_bitplane_gray_sharded(
                gr_map, resid_2d,
                profile.shard_map, profile.noise_shard_id
            )
            flag |= FLAG_BITPLANE
            header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
            final_payload = b"ZPNGCSDE" + header_base + bit_payload + metadata_bytes
        elif use_bitplane and selected_mode == "RGB":
            # [v7.3] Shard-Conditioned Bitplane RGB Path
            gr_resid = predict_2d_residuals(gr_map)
            rd_resid = predict_2d_residuals(rd_map)
            bd_resid = predict_2d_residuals(bd_map)
            bit_payload = compress_bitplane_rgb_sharded(
                gr_map, rd_map, bd_map,
                gr_resid, rd_resid, bd_resid,
                profile.shard_map, profile.noise_shard_id
            )
            flag |= FLAG_BITPLANE
            header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
            final_payload = b"ZPNGCSDE" + header_base + bit_payload + metadata_bytes
        else:
            final_payload, modes_diag = pack_bitstream(
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
                selected_mode = "PASSTHROUGH"
            elif size_simple == min_size:
                flag = (flag & FLAG_RGBA) | FLAG_SIMPLE
                header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
                final_payload = b"ZPNGCSDE" + header_base + simple_payload + metadata_bytes
                selected_mode = "SIMPLE"
            else:
                # Fallback to RAW (Uncompressed raw pixels)
                flag = (flag & FLAG_RGBA) | FLAG_RAW
                header_base = np.array([h, w, metadata_len, flag], dtype='<u4').tobytes()
                if not raw_bytes: raw_bytes = arr.tobytes()
                final_payload = b"ZPNGCSDE" + header_base + raw_bytes + metadata_bytes
                selected_mode = "RAW"

        if output_path:
            with open(output_path, 'wb') as f_out: f_out.write(final_payload)

        res_modes: npt.NDArray[np.uint8] = modes_diag

        return ZpngResult(enc_time=time.time()-t0, h=h, w=w, is_rgba=is_rgba, comp_size=len(final_payload),
                          orig_size=orig_size, hits=hits_total_p1, res_sums=sums_total_p1,
                          shard_counts=shard_counts,
                          shard_ptrs=None,
                          shard_stats=shard_stats,
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
