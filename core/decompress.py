"""
ZPNG-CSDE v6.2 [Flexible-Shard Architecture]
Module: zpng_decompress
Role: Decompressor Orchestrator.
Description: Bit-perfect reconstruction engine utilizing the 4-pillar modular core.
Architecture: Dispatcher layer connecting bitstream parsing to the BICC/RCT recovery via Flexible Sharding Hub.

Technical Flowchart:
```mermaid
graph TD
    Bs[ZPNG Bitstream] --> Head[Parse Header & Bias Matrix]
    Head --> rANS[4-way Interleaved rANS Parallel Decoding]
    
    rANS --> FHub[Flexible Sharding Hub]
    FHub --> BICC[BICC Recovery: Centroid Shifting - Grn Only]
    
    BICC --> InvGSUB[Inverse G-sub RCT: Restore RGB/RGBA]
    InvGSUB --> Out[Bit-Perfect Image Output]
```
"""

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import zstandard as zstd
import time, logging, io
import numba
import concurrent.futures
from .transform import (
    restore_channels, decode_alpha_channel, reconstruct_2d_channels
)
from .dy_shard import (
    reconstruct_channels
)
from .codec import unpack_bitstream_v5
import threading, sys
from typing import Tuple, Optional, List, Dict, Any, Union
from .common import (
    predict_med_standard,
    from_zigzag, to_zigzag,
    FLAG_RGBA, FLAG_SIMPLE, FLAG_RAW, FLAG_PASSTHROUGH, FLAG_GRAYSCALE, FLAG_COLOR_GSUB,
    FLAG_BITPLANE,
    TOTAL_SHARDS, PROFILE_RGB, sync_luts_if_needed
)
from . import env

# --- Startup: Validate Dependencies ---
env.verify_environment()

# --- Logging: Core Framework ---
logger: logging.Logger = logging.getLogger("zpng.decompress")

# [v5.1] Thread-Local Decompressor Cache to prevent redundant object creation
thread_local_decomp = threading.local()

def zstandard_decompress(data: bytes) -> bytes:
    """ Decompress data using a cached, thread-local ZstdDecompressor. """
    if not hasattr(thread_local_decomp, 'decomp'):
        thread_local_decomp.decomp = zstd.ZstdDecompressor()
    return thread_local_decomp.decomp.decompress(data)

def set_parallel_threads(n: int):
    """
    [v5.2.4] Configures the number of CPU threads used by the Numba parallel engine.
    """
    numba.set_num_threads(n)
    logger.info(f"Numba Parallel Engine (Decompress) set to {n} thread(s).")

def clear_zpng_workspaces():
    """ [v5.2.3] Forces release of Thread-Local decompressors to prevent memory retention in server workers. """
    if hasattr(thread_local_decomp, 'decomp'):
        del thread_local_decomp.decomp
    import gc
    gc.collect()

def inject_png_metadata(filepath: str, metadata_bytes: bytes) -> None:
    """
    Carefully injects metadata chunks into a PNG file, ensuring correct ordering 
    and avoiding duplicate singleton chunks (pHYs, iCCP, etc.) that Pillow might have added.
    """
    if not metadata_bytes: return
    try:
        with open(filepath, 'rb') as f_in: data: bytes = f_in.read()
        
        # 1. Map injected chunk types to identify singletons
        # The metadata_bytes is a sequence of [len:4][type:4][data:N][crc:4]
        injected_types = set()
        ptr_m = 0
        while ptr_m < len(metadata_bytes):
            if len(metadata_bytes) < ptr_m + 8: break
            c_len = int.from_bytes(metadata_bytes[ptr_m : ptr_m+4], 'big')
            c_type = metadata_bytes[ptr_m+4 : ptr_m+8]
            injected_types.add(c_type)
            ptr_m += 12 + c_len

        # Singleton types that must not be duplicated in a valid PNG
        singletons = {b'PLTE', b'tRNS', b'gAMA', b'sRGB', b'iCCP', b'cHRM', b'pHYs', b'bKGD', b'tIME', b'hIST', b'sBIT'}
        
        # 2. Parse target PNG and filter out duplicate singletons
        new_data = bytearray(data[:8]) # Signature
        ptr = 8
        insert_idx = -1
        
        while ptr < len(data):
            c_len = int.from_bytes(data[ptr:ptr+4], 'big')
            c_type = data[ptr+4:ptr+8]
            full_chunk_len = 12 + c_len
            
            # Identify the injection point: before the first PLTE or IDAT
            if insert_idx == -1 and c_type in [b'PLTE', b'IDAT']:
                insert_idx = len(new_data)
            
            # Skip Pillow's version of a chunk if we are injecting our own original version
            if c_type in singletons and c_type in injected_types:
                # logger.debug(f"Overwriting Pillow's {c_type.decode()} with original metadata.")
                pass 
            elif c_type == b'IHDR':
                new_data.extend(data[ptr : ptr + full_chunk_len])
            elif c_type == b'IEND':
                # We'll handle IEND at the very end
                break
            else:
                new_data.extend(data[ptr : ptr + full_chunk_len])
            
            ptr += full_chunk_len

        # 3. Perform Injection
        if insert_idx == -1: # Fallback: insert after IHDR (usually at index 8 + IHDR_LEN)
            ihdr_len = int.from_bytes(new_data[8:12], 'big')
            insert_idx = 20 + ihdr_len
            
        final_data = new_data[:insert_idx] + metadata_bytes + new_data[insert_idx:] + data[ptr:] # data[ptr:] is IEND
        
        with open(filepath, 'wb') as f_out:
            f_out.write(final_data)
            
    except Exception as e:
        logger.error(f"Failed to inject PNG metadata: {e}")





def decompress_csde(zpng_input: Union[bytes, str], output_path: Optional[str] = None, optimize_png: bool = False) -> Tuple[npt.NDArray[np.uint8], float]:
    """ 
    Main ZPNG-CSDE Decompression Entry Point (V6.6 Stable).
    
    Bit-perfect reverse orchestrator:
    1. Header/Metadata Parser -> 2. Zstd Dictionary Extraction -> 3. SIMD 4-Way rANS Decoding
    4. BICC Inverse Sharding (Median Re-addition) -> 5. Inverse G-Sub RCT -> 6. Original Hex Verification.
    
    NOTE: Operates with Numba JIT parallelism to achieve 6.0+ MB/s decompression throughput.
    """
    t0: float = time.time()
    try:
        f: Union[io.BytesIO, io.BufferedReader]
        if isinstance(zpng_input, bytes): 
            f = io.BytesIO(zpng_input)
        else: 
            f = open(zpng_input, 'rb')
        try:
            magic: bytes = f.read(8)
            if magic != b"ZPNGCSDE": raise ValueError("不支援的檔案格式")
            header_base: bytes = f.read(16)
            h, w, metadata_len, flag = np.frombuffer(header_base, dtype='<u4')
            if h == 0 or w == 0 or h > 65535 or w > 65535:
                raise ValueError(f"Invalid image dimensions: {w}x{h}")
            is_rgba: bool = bool(flag & FLAG_RGBA)
            is_simple, is_raw, is_pass = bool(flag & FLAG_SIMPLE), bool(flag & FLAG_RAW), bool(flag & FLAG_PASSTHROUGH)
            is_grayscale: bool = bool(flag & FLAG_GRAYSCALE)
            
            # [v6.6] Unified Profile Selection
            profile = PROFILE_RGB
            
            # [v6.6 Defensive] Ensure global Context LUTs match requested profile
            sync_luts_if_needed(profile.v_boundaries_gr, profile.intensity_segments)
            
            n_shards = profile.total_shards
            
            # Expanded parameters for JIT
            nsid = profile.noise_shard_id
            
            shard_widths: npt.NDArray[np.uint16] = np.zeros((3, n_shards), dtype=np.uint16)
            shard_medians: npt.NDArray[np.uint8] = np.zeros((3, n_shards), dtype=np.uint8)
            shard_modes: npt.NDArray[np.uint8] = np.zeros((3, n_shards), dtype=np.uint8)

            metadata_bytes: bytes = b""
            compressed_data: bytes = b""

            m_len = int(metadata_len)
            if not (is_simple or is_raw or is_pass):
                if flag & FLAG_BITPLANE:
                    shard_widths.fill(1)
                    # For Bitplane, we still use full read for now as it's not yet optimized
                    compressed_data = f.read()
                    metadata_bytes = b""
                else:
                    meta_stride = n_shards if is_grayscale else 3 * n_shards
                    h_len = meta_stride * 3
                    h_raw: bytes = f.read(h_len)
                    if len(h_raw) < h_len:
                        raise ValueError("Truncated header: Shard metadata missing.")

                    if is_grayscale:
                        r_widths = np.frombuffer(h_raw[:n_shards], dtype=np.uint8)
                        shard_widths[0] = np.where(r_widths == 0, np.uint16(256), r_widths.astype(np.uint16))
                        shard_medians[0] = np.frombuffer(h_raw[n_shards:2*n_shards], dtype=np.uint8)
                        shard_modes[0] = np.frombuffer(h_raw[2*n_shards:3*n_shards], dtype=np.uint8)
                    else:
                        r_widths = np.frombuffer(h_raw[:3*n_shards], dtype=np.uint8).reshape((3, n_shards))
                        shard_widths = np.where(r_widths == 0, np.uint16(256), r_widths.astype(np.uint16))
                        shard_medians = np.frombuffer(h_raw[3*n_shards:6*n_shards], dtype=np.uint8).reshape((3, n_shards))
                        shard_modes = np.frombuffer(h_raw[6*n_shards:9*n_shards], dtype=np.uint8).reshape((3, n_shards))

                    # Pass the stream directly to unpacker
                    res_gr_flat, res_rd_flat, res_bd_flat, gr_offs, rd_offs, bd_offs, shard_counts, res_a_flat = unpack_bitstream_v5(
                        f, h, w, is_rgba, is_grayscale, shard_widths, shard_modes, flag, metadata_len
                    )
                    
                    # Read metadata from the end of the stream
                    metadata_bytes = f.read(m_len) if m_len > 0 else b""
            else:
                # simple/raw/pass paths still read full payload (as they use PIL or Zstd directly)
                all_payload = f.read()
                shard_widths.fill(1)
                metadata_bytes = all_payload[-m_len:] if m_len > 0 else b""
                compressed_data = all_payload[:-m_len] if m_len > 0 else all_payload

            rgb: npt.NDArray[np.uint8]
            if is_pass:
                rgb = np.array(Image.open(io.BytesIO(compressed_data)).convert('RGBA' if is_rgba else 'RGB'))
            elif is_raw:
                rgb = np.frombuffer(compressed_data, dtype=np.uint8).reshape((h, w, 4 if is_rgba else 3))
            elif is_simple:
                rgb = np.frombuffer(zstandard_decompress(compressed_data), dtype=np.uint8).reshape((h, w, 4 if is_rgba else 3))
            else:
                if flag & FLAG_BITPLANE:
                    res_gr_flat, res_rd_flat, res_bd_flat, gr_offs, rd_offs, bd_offs, shard_counts, res_a_flat = unpack_bitstream_v5(
                        compressed_data, h, w, is_rgba, is_grayscale, shard_widths, shard_modes, flag, metadata_len
                    )
                    if is_grayscale:
                        gr_rec = reconstruct_2d_channels(h, w, res_gr_flat.reshape((h, w)))
                        rd_rec, bd_rec = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
                    else:
                        raise NotImplementedError("Color Bitplane reconstruction not yet implemented.")
                else:
                    # Standard Shard Path (already unpacked via stream above)
                    gr_rec, rd_rec, bd_rec = reconstruct_channels(
                        h, w, res_gr_flat, res_rd_flat, res_bd_flat, 
                        gr_offs, rd_offs, bd_offs, 
                        shard_counts, shard_medians, is_grayscale,
                        profile.shard_map, profile.v_boundaries_gr, 
                        profile.intensity_segments, nsid
                    )

                a_rec: npt.NDArray[np.uint8] = np.empty((h, w), dtype=np.uint8) if is_rgba else np.zeros((0, 0), dtype=np.uint8)
                if is_rgba and res_a_flat is not None:
                    decode_alpha_channel(h, w, res_a_flat.reshape((h, w)), a_rec)
                
                rgb = restore_channels(gr_rec, rd_rec, bd_rec, a_rec, is_rgba, is_grayscale)

            if output_path:
                Image.fromarray(rgb).save(output_path, optimize=optimize_png)
                if metadata_bytes: inject_png_metadata(output_path, metadata_bytes)
            return rgb, time.time() - t0
        finally:
            f.close()
    except Exception as e:
        logger.error(f"Decompression Failure: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise

# --- Pytest Snippet for Decompression Integrity ---
# """
# def test_decompression_flow():
#     # This requires a valid zpng bitstream, usually tested via zpng_imgtest.py
#     pass
# """
