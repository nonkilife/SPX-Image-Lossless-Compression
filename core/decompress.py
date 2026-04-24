"""
SPX v8.3.2 [Stable Parallel Architecture]
Module: spx_decompress
Role: Decompressor Orchestrator.
Description: Bit-perfect reconstruction engine utilizing the 4-pillar modular core.
Architecture: Dispatcher layer connecting bitstream parsing to the BICC/RCT recovery (Pillars 2 & 3) via Parallel Shard Recombination.

Technical Flowchart:
```mermaid
graph TD
    In[SPX Bitstream] --> Flags{Mode Flags}
    Flags -->|PASSTHROUGH| Pass[PIL Direct Open]
    Flags -->|RAW| Raw[memcpy reshape]
    Flags -->|SIMPLE| Simple[ZStd Decompress]
    Flags -->|Sharded| BP{Is Bitplane?}
    BP -->|No| Standard[Standard rANS Decoding]
    BP -->|Yes| Bitplane[Bitplane rANS Decoding]
    Standard & Bitplane --> BICC[BICC: Median Re-addition]
    BICC --> InvGSUB[Inverse G-sub RCT: Restore RGB]
    Pass & Raw & Simple & InvGSUB --> Out[Bit-Perfect Image]
```
"""

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import zstandard as zstd
import time, logging, io
import numba
from .transform import (
    restore_channels, decode_alpha_channel, reconstruct_2d_channels
)
from .sharding import PROFILE_RGB, reconstruct_shards_rgb
from .codec import unpack_bitstream
from .rans_bitplane import decompress_bitplane_rgb_sharded, decompress_bitplane_gray_sharded
import threading
from typing import Tuple, Optional, Union
from .common import (
    FLAG_RGBA, FLAG_SIMPLE, FLAG_RAW, FLAG_PASSTHROUGH, FLAG_GRAYSCALE,
    FLAG_BITPLANE, FLAG_COLOR_GSUB
)
from .sharding import PROFILE_RGB
from . import env

# --- Startup: Validate Dependencies ---
env.verify_environment()

# --- Logging: Core Framework ---
logger: logging.Logger = logging.getLogger("spx.decompress")

# [v8.3.2] Thread-Local Decompressor Cache to prevent redundant object creation
thread_local_decomp = threading.local()

def zstandard_decompress(data: bytes) -> bytes:
    """ Decompress data using a cached, thread-local ZstdDecompressor. """
    if not hasattr(thread_local_decomp, 'decomp'):
        thread_local_decomp.decomp = zstd.ZstdDecompressor()
    return thread_local_decomp.decomp.decompress(data)

def set_parallel_threads(n: int):
    """
    [v8.3.2] Configures the number of CPU threads used by the Numba parallel engine.
    """
    numba.set_num_threads(n)
    logger.info(f"Numba Parallel Engine (Decompress) set to {n} thread(s).")

def clear_spx_workspaces():
    """ [v8.3.2] Forces release of Thread-Local decompressors to prevent memory retention in server workers. """
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


def decompress_spx(spx_input: Union[bytes, str], output_path: Optional[str] = None, optimize_png: bool = False) -> Tuple[npt.NDArray[np.uint8], float]:
    """
    Main SPX Decompression Entry Point (v8.3.2 Stable).

    Bit-perfect reverse orchestrator:
    1. BICC Shard Recovery (Pillar 2): Staggered reconstruction of G-Lead then RD/BD-Lag.
    2. Inverse GSUB (Pillar 1): Cross-channel restoration using Green as the spatial reference.

    NOTE: Operates with Numba JIT parallelism to achieve 25.0+ MB/s decompression throughput.
    """
    t0: float = time.time()
    try:
        f: Union[io.BytesIO, io.BufferedReader]
        if isinstance(spx_input, bytes): 
            f = io.BytesIO(spx_input)
        else:
            f = open(spx_input, 'rb')
        try:
            # 1. Unified Bitstream Unpacking (Handles Magic, Header, Shards, and Metadata)
            unpacked = unpack_bitstream(f, PROFILE_RGB)
            h, w, flag = unpacked.h, unpacked.w, unpacked.flag
            is_rgba: bool = bool(flag & FLAG_RGBA)
            is_simple, is_raw, is_pass = bool(flag & FLAG_SIMPLE), bool(flag & FLAG_RAW), bool(flag & FLAG_PASSTHROUGH)
            is_grayscale: bool = bool(flag & FLAG_GRAYSCALE)
            
            metadata_bytes: bytes = unpacked.metadata
            res_gr_flat, res_rd_flat, res_bd_flat = unpacked.res_gr, unpacked.res_rd, unpacked.res_bd
            gr_offs, rd_offs, bd_offs = unpacked.gr_offs, unpacked.rd_offs, unpacked.bd_offs
            shard_counts = unpacked.shard_counts
            res_a_flat = unpacked.res_a
            
            profile = PROFILE_RGB
            nsid = profile.noise_shard_id

            # [v8.3.2] Universal Payload Handling
            compressed_data: bytes = unpacked.payload
            
            rgb: npt.NDArray[np.uint8]
            if is_pass:
                rgb = np.array(Image.open(io.BytesIO(compressed_data)).convert('RGBA' if is_rgba else 'RGB'))
            elif is_raw:
                if is_grayscale:
                    rgb = np.frombuffer(compressed_data, dtype=np.uint8).reshape((h, w))
                else:
                    rgb = np.frombuffer(compressed_data, dtype=np.uint8).reshape((h, w, 4 if is_rgba else 3))
            elif is_simple:
                if is_grayscale:
                    rgb = np.frombuffer(zstandard_decompress(compressed_data), dtype=np.uint8).reshape((h, w))
                else:
                    rgb = np.frombuffer(zstandard_decompress(compressed_data), dtype=np.uint8).reshape((h, w, 4 if is_rgba else 3))
            else:
                if flag & FLAG_BITPLANE:
                    if is_grayscale:
                        # [v8.3.2] Grayscale bitplane now decoded here for protocol symmetry
                        res_gr_raw, ptr_bp = decompress_bitplane_gray_sharded(compressed_data, h, w, profile)
                        gr_rec = reconstruct_2d_channels(h, w, res_gr_raw.reshape((h, w)))
                        rd_rec = np.zeros((h, w), dtype=np.uint8)
                        bd_rec = np.zeros((h, w), dtype=np.uint8)
                        
                        if is_rgba:
                            # Correct Alpha extraction logic for bitplane grayscale
                            a_len = int(np.frombuffer(compressed_data[ptr_bp : ptr_bp+4], dtype=np.uint32)[0])
                            ptr_bp += 4
                            res_a_flat = np.frombuffer(zstandard_decompress(compressed_data[ptr_bp : ptr_bp+a_len]), dtype=np.uint8)
                    else:
                        gr_rec, rd_rec, bd_rec, ptr = decompress_bitplane_rgb_sharded(
                            compressed_data, h, w, profile
                        )
                        if is_rgba:
                            # res_a_flat logic for RGB bitplane
                            a_len = int(np.frombuffer(compressed_data[ptr:ptr+4], dtype=np.uint32)[0])
                            ptr += 4
                            res_a_flat = np.frombuffer(zstandard_decompress(compressed_data[ptr:ptr+a_len]), dtype=np.uint8)
                else:
                    # Standard Shard Path
                    gr_rec, rd_rec, bd_rec = reconstruct_shards_rgb(
                        h, w, res_gr_flat, res_rd_flat, res_bd_flat,
                        gr_offs, rd_offs, bd_offs, is_grayscale,
                        profile.spatial_lut, profile.intensity_lut, profile.dispatch_lut
                    )

                a_rec: npt.NDArray[np.uint8] = np.zeros((h, w), dtype=np.uint8) if is_rgba else np.zeros((0, 0), dtype=np.uint8)
                if is_rgba and res_a_flat is not None:
                    decode_alpha_channel(h, w, res_a_flat.reshape((h, w)), a_rec)
                
                rgb = restore_channels(gr_rec, rd_rec, bd_rec, a_rec, is_rgba, is_grayscale,
                                      bool(flag & FLAG_COLOR_GSUB))

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

