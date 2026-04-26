r"""
SPX v8.3.2 Unified Benchmark Suite (test_suite)
Module: test_suite
Role: QA & Performance Validation.

Description: 
Cross-codec comparative benchmarking and bit-perfect verification (MSE). This 
module serves as the authoritative validation hub for SPX performance against 
industry standards like WebP and JPEG-XL.

Architecture & Engineering Rationale:
1. Throughput-Weighted Timing: Reporting aggregate core-seconds often misleads 
   on many-core systems. This suite distributes the measured wall-clock time 
   proportionally based on core-load fractions to provide an accurate "Per-Image" 
   latency that reflects real-world system throughput.
2. Warmup Phase: To eliminate JIT compilation spikes (Rust) and library
   initialization overhead from the final metrics, a dedicated single-threaded 
   warmup pass is executed before the main benchmark.
3. Win Counting: Beyond aggregate averages, the suite tracks "Wins" at the 
   individual image level, identifying which codec provides the best 
   compression or speed for specific image characteristics.
4. Dataset Curation: Includes a 'Reclassification' system that segments 
   large datasets (like DIV2K) into Easy, Hard, and Hell categories based on 
   observed compression ratios, facilitating targeted optimization.

Technical Flowchart:
```mermaid
graph TD
    Input[Dataset Directory] --> Glob[File Discovery: PNG, BMP, PNM]
    Glob --> Workers[Parallel Workers: SPX, WebP, JXL]
    
    subgraph "Worker Loop"
        W1[Preload Array] --> W2[Compress: Speed + BPP]
        W2 --> W3[Decompress: Speed]
        W3 --> W4[Verify: MSE = 0]
    end
    
    Workers --> Stats[Aggregate Stats: MB/s, BPP, Savings]
    Stats --> Table[ASCII Comparison Table]
    Stats --> CSV[test_results.csv]
```
"""

__version__ = "8.3.2"
import os
import time
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
import csv
import datetime
import shutil
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import List, Dict, Tuple, Any, Optional

# Ensure core engine components are accessible
try:
    # [v8.3.2 Migration] test_suite moved inside core package
    from .compress import compress_spx
    from .decompress import decompress_spx
except ImportError:
    # Fallback for direct execution if needed
    try:
        from compress import compress_spx
        from decompress import decompress_spx
    except ImportError as e:
        print(f"[!] Error: Core modules not found: {e}")
        sys.exit(1)

# =============================================================================
# --- Global Benchmark Settings ---
# =============================================================================
WEBP_METHOD = 6
JXL_EFFORT = 7

# =============================================================================
# --- Helper Utilities ---
# =============================================================================

def get_pnm_stats(arr: npt.NDArray) -> int:
    """Calculate PNM size (Raw Pixels + Header) in bytes."""
    channels = arr.shape[2] if arr.ndim == 3 else 1
    pnm_header = f"P{'6' if channels==3 else '5'}\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode('ascii')
    return len(pnm_header) + arr.size

def get_ref_png_size(path: str) -> Optional[int]:
    if path == "__WARMUP__": return None
    base, ext = os.path.splitext(path)
    if ext.lower() == ".png":
        return os.path.getsize(path)
    # Check if a .png sibling exists
    png_path = base + ".png"
    if os.path.exists(png_path):
        return os.path.getsize(png_path)
    return None

def calculate_mse(arr1: npt.NDArray, arr2: npt.NDArray) -> float:
    """Calculate Mean Squared Error between two arrays with shape validation."""
    if arr1.shape != arr2.shape:
        # Standardize to 3D for comparison if one is grayscale
        if arr1.ndim == 2: arr1 = np.expand_dims(arr1, -1)
        if arr2.ndim == 2: arr2 = np.expand_dims(arr2, -1)
        
        # Handle grayscale (1-ch) vs RGB (3-ch) conversion by codecs
        if arr1.shape[2] == 1 and arr2.shape[2] == 3:
            arr1 = np.concatenate([arr1, arr1, arr1], axis=-1)
        elif arr1.shape[2] == 3 and arr2.shape[2] == 1:
            arr2 = np.concatenate([arr2, arr2, arr2], axis=-1)

        if arr1.shape != arr2.shape:
            return 9999.0
            
    if arr1.ndim == 3:
        c = min(arr1.shape[2], 3)
        a1 = arr1[..., :c].astype(np.float64)
        a2 = arr2[..., :c].astype(np.float64)
    else:
        a1 = arr1.astype(np.float64)
        a2 = arr2.astype(np.float64)
    return float(np.mean((a1 - a2)**2))

# =============================================================================
# --- Unified Codec Workers ---
# =============================================================================

def spx_worker(path: str, **kwargs) -> Dict[str, Any]:
    filename = os.path.basename(path)
    try:
        if path == "__WARMUP__":
            # Minimal warmup task
            d = np.zeros((64, 64, 3), dtype=np.uint8)
            res = compress_spx(None, None, preloaded_arr=d, use_bitplane=kwargs.get('use_bitplane', None))
            _, _ = decompress_spx(res.payload, None)
            return {"success": False} # Don't record warmup

        with Image.open(path) as img:
            img.load()
            # [v8.3.2] Detect optimal mode to avoid inflating baseline sizes
            target_mode = 'RGB'
            if img.mode in ('L', '1'): target_mode = 'L'
            elif img.mode == 'RGBA': target_mode = 'RGBA'
            
            arr_orig = np.array(img.convert(target_mode))
            pixels = arr_orig.shape[0] * arr_orig.shape[1]
            orig_size_bytes = os.path.getsize(path)

        t0 = time.perf_counter()
        # [v8.3.2] Propagation of bitplane flag if provided via kwargs. 
        # Default to None to enable Auto-Selection.
        use_bitplane = kwargs.get('use_bitplane', None)
        res_spx = compress_spx(path, None, preloaded_arr=arr_orig, use_bitplane=use_bitplane)
        ze_s = (time.perf_counter() - t0)
        
        t1 = time.perf_counter()
        rec_rgb, _ = decompress_spx(res_spx.payload, None, optimize_png=False)
        zd_s = (time.perf_counter() - t1)
        
        comp_size_bytes = len(res_spx.payload)
        mse = calculate_mse(arr_orig, rec_rgb)
        pnm_bytes = get_pnm_stats(arr_orig)
        png_bytes = get_ref_png_size(path)

        return {
            "name": "SPX", "filename": filename, "pixels": pixels,
            "orig_bytes": orig_size_bytes, "pnm_bytes": pnm_bytes, 
            "png_bytes": png_bytes, "comp_bytes": comp_size_bytes,
            "e_s": ze_s, "d_s": zd_s, "mse": mse, "success": True
        }
    except Exception as e:
        return {"name": "SPX", "filename": filename, "success": False, "error": str(e)}

def webp_worker(path: str, **kwargs) -> Dict[str, Any]:
    filename = os.path.basename(path)
    try:
        if path == "__WARMUP__": return {"success": False}
        with Image.open(path) as img:
            target_mode = 'RGB'
            if img.mode in ('L', '1'): target_mode = 'L'
            elif img.mode == 'RGBA': target_mode = 'RGBA'
            img = img.convert(target_mode)
            arr_orig = np.array(img)
            pixels = arr_orig.shape[0] * arr_orig.shape[1]
            orig_size_bytes = os.path.getsize(path)
            
            buffer = BytesIO()
            t0 = time.perf_counter()
            img.save(buffer, format='WEBP', lossless=True, quality=100, method=WEBP_METHOD)
            we_s = (time.perf_counter() - t0)
            
            comp_size_bytes = buffer.getbuffer().nbytes
            buffer.seek(0)
            
            t1 = time.perf_counter()
            with Image.open(buffer) as img_webp:
                img_webp.load()
                arr_rec = np.array(img_webp) # Force full decode
            wd_s = (time.perf_counter() - t1)
            
            pnm_bytes = get_pnm_stats(arr_orig)
            mse = calculate_mse(arr_orig, arr_rec)
            png_bytes = get_ref_png_size(path)

            return {
                "name": "WebP", "filename": filename, "pixels": pixels,
                "orig_bytes": orig_size_bytes, "pnm_bytes": pnm_bytes, 
                "png_bytes": png_bytes, "comp_bytes": comp_size_bytes,
                "e_s": we_s, "d_s": wd_s, "mse": mse, "success": True
            }
    except Exception as e:
        return {"name": "WebP", "filename": filename, "success": False, "error": str(e)}

def jxl_worker(path: str, **kwargs) -> Dict[str, Any]:
    filename = os.path.basename(path)
    try:
        import imagecodecs
    except ImportError:
        return {"name": "JXL", "filename": filename, "success": False, "error": "imagecodecs not installed"}

    try:
        if path == "__WARMUP__": return {"success": False}
        with Image.open(path) as img:
            target_mode = 'RGB'
            if img.mode in ('L', '1'): target_mode = 'L'
            elif img.mode == 'RGBA': target_mode = 'RGBA'
            img = img.convert(target_mode)
            arr_orig = np.array(img)
            pixels = arr_orig.shape[0] * arr_orig.shape[1]
            orig_size_bytes = os.path.getsize(path)
            
            t0 = time.perf_counter()
            encoded = imagecodecs.jpegxl_encode(arr_orig, level=JXL_EFFORT, distance=0)
            je_s = (time.perf_counter() - t0)
            
            t1 = time.perf_counter()
            arr_rec = imagecodecs.jpegxl_decode(encoded)
            jd_s = (time.perf_counter() - t1)
            
            comp_size_bytes = len(encoded)
            pnm_bytes = get_pnm_stats(arr_orig)
            mse = calculate_mse(arr_orig, arr_rec)
            png_bytes = get_ref_png_size(path)

            return {
                "name": "JXL", "filename": filename, "pixels": pixels,
                "orig_bytes": orig_size_bytes, "pnm_bytes": pnm_bytes, 
                "png_bytes": png_bytes, "comp_bytes": comp_size_bytes,
                "e_s": je_s, "d_s": jd_s, "mse": mse, "success": True
            }
    except Exception as e:
        return {"name": "JXL", "filename": filename, "success": False, "error": str(e)}

# =============================================================================
# --- Unified Reporter ---
# =============================================================================

class CodecReporter:
    def __init__(self, codec_name: str):
        self.name = codec_name
        self.count = 0
        self.total_orig = 0
        self.total_pnm = 0
        self.total_png = 0
        self.png_count = 0
        self.total_comp = 0
        self.total_pixels = 0
        self.total_e_s = 0.0
        self.total_d_s = 0.0
        self.total_mse = 0.0
        self.ratios = []

    def record(self, res: Dict):
        if not res or not res.get("success"): return
        self.count += 1
        self.total_pixels += res["pixels"]
        self.total_orig += res["orig_bytes"]
        self.total_pnm += res["pnm_bytes"]
        if res.get("png_bytes") is not None:
            self.total_png += res["png_bytes"]
            self.png_count += 1
        self.total_comp += res["comp_bytes"]
        self.total_e_s += res["e_s"]
        self.total_d_s += res["d_s"]
        self.total_mse += (res["mse"] * res["pixels"])
        self.ratios.append(res["comp_bytes"] / res["orig_bytes"] * 100)

    def get_stats(self, wall_clock: float) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "name": self.name, "count": 0, "success": False,
                "orig_mb": 0, "pnm_mb": 0, "pnm_bpp": 0, "src_bpp": 0, 
                "png_mb": None, "png_bpp": None,
                "comp_mb": 0, "saved_pnm_pct": 0, "saved_pct": 0, "saved_png_pct": None,
                "bpp": 0, "mean_ratio": 0, "median_ratio": 0,
                "range": (0, 0), "mse": 0, "avg_e_ms": 0, "avg_d_ms": 0, "wall_s": wall_clock,
                "sys_tp": (0, 0), "total_orig": 0, "total_pixels": 0, "core_tp": (0, 0)
            }
        
        agg_ratio = self.total_comp / self.total_orig * 100
        mean_ratio = np.mean(self.ratios)
        avg_bpp = self.total_comp * 8.0 / self.total_pixels
        src_bpp = self.total_orig * 8.0 / self.total_pixels if self.total_pixels > 0 else 0
        orig_mb = self.total_orig / (1024**2)

        total_work = self.total_e_s + self.total_d_s
        if total_work > 0:
            # Normalized Wall-Clock: Distribute wall-clock proportionally to core-second fractions.
            enc_wall = wall_clock * (self.total_e_s / total_work)
            dec_wall = wall_clock * (self.total_d_s / total_work)
            sys_enc_tp = orig_mb / enc_wall if enc_wall > 0 else 0
            sys_dec_tp = orig_mb / dec_wall if dec_wall > 0 else 0
        else:
            enc_wall = dec_wall = sys_enc_tp = sys_dec_tp = 0

        return {
            "name": self.name,
            "count": self.count,
            "orig_mb": orig_mb,
            "pnm_mb": self.total_pnm / (1024**2),
            "pnm_bpp": self.total_pnm * 8.0 / self.total_pixels if self.total_pixels > 0 else 0,
            "src_bpp": src_bpp,
            "png_mb": self.total_png / (1024**2) if self.png_count > 0 else None,
            "png_bpp": self.total_png * 8.0 / self.total_pixels if self.png_count > 0 and self.total_pixels > 0 else None,
            "comp_mb": self.total_comp / (1024**2),
            "saved_pnm_pct": (1.0 - self.total_comp / self.total_pnm) * 100 if self.total_pnm > 0 else 0,
            "saved_png_pct": (1.0 - self.total_comp / self.total_png) * 100 if self.png_count > 0 and self.total_png > 0 else None,
            "saved_pct": 100 - agg_ratio, "bpp": avg_bpp,
            "mean_ratio": mean_ratio, "median_ratio": np.median(self.ratios),
            "range": (min(self.ratios), max(self.ratios)),
            "mse": self.total_mse / self.total_pixels,
            # [v8.3.2 Fix] Reporting Throughput-weighted time per image instead of aggregate core-seconds.
            "avg_e_ms": (enc_wall * 1000) / self.count if self.count > 0 else 0,
            "avg_d_ms": (dec_wall * 1000) / self.count if self.count > 0 else 0,
            "enc_wall_s": enc_wall,
            "dec_wall_s": dec_wall,
            "wall_s": wall_clock,
            "sys_tp": (sys_enc_tp, sys_dec_tp),
            "total_orig": self.total_orig,
            "total_pixels": self.total_pixels,
            "core_tp": (orig_mb / self.total_e_s if self.total_e_s > 0 else 0, orig_mb / self.total_d_s if self.total_d_s > 0 else 0)
        }

# [AI DIRECTIVE] The following table format (print_comparison_table) and 
# metric ordering are strictly optimized for the user's research workflow. 
# DO NOT truncate, reorder, or modify this output format/logic without 
# explicit user confirmation. Preserve all division lines (SEPARATORs).
def print_comparison_table(stats_list: List[Dict], dataset_name: str):
    if not stats_list or not any(stats_list): return
    valid_stats = [s for s in stats_list if s]
    img_count = valid_stats[0].get('count', 0)
    
    # Headers
    top_left = dataset_name
    headers = [top_left] + [s['name'] for s in valid_stats]
    col_widths = [max(len(top_left), 25)] + [max(len(h), 18) for h in headers[1:]]
    
    # Define Rows (Label, key, format, tuple_idx)
    rows = [
        ("PNM Size", "pnm_mb", "{:>10.2f} MB"),
        (f"Dataset Size ({int(img_count)} imgs)", "orig_mb", "{:>10.2f} MB"),
        ("SPX Size", "comp_mb", "{:>10.2f} MB"),
        ("BPP (PNM)", "pnm_bpp", "{:>13.4f}"),
        ("BPP (PNG)", "png_bpp", "{:>13.4f}"),
        ("BPP (Compressed)", "bpp", "{:>13.6f}"),
        ("SEPARATOR", "", ""),
        ("Savings % (vs PNM)", "saved_pnm_pct", "{:>10.2f} %"),
        ("Savings % (vs PNG)", "saved_png_pct", "{:>10.2f} %"),
        ("Mean Ratio (%)", "mean_ratio", "{:>10.2f} %"),
        ("Median Ratio (%)", "median_ratio", "{:>10.2f} %"),
        ("Ratio Range (%)", "range", "{0:5.1f}-{1:1.1f} %"),
        ("SEPARATOR", "", ""),
        ("Avg Enc Time", "avg_e_ms", "{:>10.1f} ms"),
        ("Avg Dec Time", "avg_d_ms", "{:>10.1f} ms"),
        ("Total Enc Time", "enc_wall_s", "{:>10.2f} s"),
        ("Total Dec Time", "dec_wall_s", "{:>10.2f} s"),
        ("Warmup Time", "warmup_s", "{:>10.2f} s"),
        ("Wall-clock", "wall_s", "{:>10.2f} s"),
        ("SEPARATOR", "", ""),
        ("Single Core (Enc)", "core_tp", "{:>10.2f} MB/s", 0),
        ("Single Core (Dec)", "core_tp", "{:>10.2f} MB/s", 1),
        ("Throughput (Enc)", "sys_tp", "{:>10.2f} MB/s", 0),
        ("Throughput (Dec)", "sys_tp", "{:>10.2f} MB/s", 1),
        ("SEPARATOR", "", ""),
        ("Wins: Space", "wins_s", "{:^13d}"),
        ("Wins: Encode", "wins_e", "{:^13d}"),
        ("Wins: Decode", "wins_d", "{:^13d}"),
        ("MSE (Quality)", "mse", "{:>13.8f}")
    ]
    
    total_width = sum(col_widths) + (len(headers) * 3) + 1
    print("\n" + "=" * total_width)
    
    # Header
    header_str = "|"
    sep_str = "|"
    for i, h in enumerate(headers):
        if i == 0:
            header_str += f" {h:<{col_widths[i]}} |"
        else:
            header_str += f" {h:^{col_widths[i]}} |"
        sep_str += "-" * (col_widths[i] + 2) + "|"
    print(header_str)
    print(sep_str)
    
    # Rows
    for r_label, key, fmt, *idx in rows:
        if r_label == "SEPARATOR":
            print(sep_str)
            continue
            
        row_str = f"| {r_label:<{col_widths[0]}} |"
        for i, s in enumerate(valid_stats):
            val = s.get(key, "-")
            if idx and isinstance(val, (tuple, list)):
                val = val[idx[0]]
            
            if val == "-" or val is None:
                formatted_val = f"{'n/a':^{col_widths[i+1]}}"
            else:
                try:
                    if isinstance(val, (tuple, list)):
                        formatted_val = fmt.format(*val)
                    else:
                        formatted_val = fmt.format(val)
                except Exception:
                    formatted_val = f"{'ERR':>{col_widths[i+1]}}"
                
                formatted_val = formatted_val[:col_widths[i+1]]
            
            row_str += f" {formatted_val:<{col_widths[i+1]}} |"
        print(row_str)
    
    # Bottom border
    print("=" * total_width + "\n")

def show_codec_summary(s: Dict):
    if not s or 'name' not in s: return
    div = "-" * 70
    print(f"\n{div}")
    print(f"  {s['name']} Performance Audit ({int(s.get('count',0))} images):")
    print(f"  PNM Size      : {s['pnm_mb']:6.2f} MB | BPP {s['pnm_bpp']:6.4f}")
    print(f"  Dataset Size  : {s['orig_mb']:6.2f} MB | BPP {s['src_bpp']:6.4f}")
    print(f"  SPX Size      : {s['comp_mb']:6.2f} MB | BPP {s['bpp']:6.4f}")
    png_saved = f"{s['saved_png_pct']:5.2f}%" if s.get('saved_png_pct') is not None else "n/a"
    print(f"  Savings %     : vs PNM {s['saved_pnm_pct']:5.2f}% | vs PNG {png_saved}")
    print(f"  Comp. Ratio   : Mean {s['mean_ratio']:5.2f}% | Median {s['median_ratio']:5.2f}% | Range {s['range'][0]:5.1f}-{s['range'][1]:5.1f}%")
    print(f"  Avg File Speed: Enc {s['avg_e_ms']:7.1f} ms | Dec {s['avg_d_ms']:7.1f} ms")
    print(f"  Total Phase   : Enc {s['enc_wall_s']:7.2f} s  | Dec {s['dec_wall_s']:7.2f} s")
    print(f"  Throughput    : Enc {s['sys_tp'][0]:6.2f} MB/s | Dec {s['sys_tp'][1]:6.2f} MB/s")
    print(f"  Single Core   : Enc {s['core_tp'][0]:6.2f} MB/s | Dec {s['core_tp'][1]:6.2f} MB/s")
    print(f"  Wall-clock    : {s['wall_s']:6.2f} s")
    print(f"  MSE (Quality) : {s['mse']:13.8f}")
    print(div)

# =============================================================================
# --- Main Orchestrator ---
# =============================================================================

def run_codec_benchmark(codec_name: str, worker_fn: Any, files: List[str], workers: int, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print(f"[*] Benchmarking {codec_name}...")
    reporter = CodecReporter(codec_name)
    results = {}
    
    try:
        # --- [Critical Fix 1]: Isolated Single-Threaded Warmup ---
        msg_warmup = f"  [Warmup] Compiling (Single Thread)..."
        print(msg_warmup, end='\r', flush=True)
        w_start = time.perf_counter()
        worker_fn("__WARMUP__", **kwargs)
        t_warmup = time.perf_counter() - w_start
        # Build completion message that fully overwrites the warmup string
        done_msg = f"  [Warmup] Completed in {t_warmup:.2f}s"
        print(done_msg.ljust(len(msg_warmup) + 5))

        # --- [Critical Fix 2]: Parallelized Stress Testing ---
        t_proc_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker_fn, p, **kwargs): p for p in files}
            idx = 0
            for f in as_completed(futures):
                idx += 1
                res = f.result()
                reporter.record(res)
                if res.get("success"):
                    results[res["filename"]] = res
                else:
                    print(f"\n  [!] Error in {codec_name} for {res.get('filename')}: {res.get('error')}")
                if idx % 10 == 0 or idx == len(files):
                    prog_msg = f"  Progress: {idx}/{len(files)} processed"
                    print(f"\r{prog_msg}", end='', flush=True)
            print()
    except KeyboardInterrupt:
        print("\n[!] Aborted.")
        raise
    except Exception as e:
        print(f"\n[!] FATAL ERROR in run_codec_benchmark: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    wall_clock = time.perf_counter() - t_proc_start   # excludes warmup time
    stats = reporter.get_stats(wall_clock)
    stats["warmup_s"] = t_warmup
    show_codec_summary(stats)
    return stats, results

def export_to_csv(stats_list: List[Dict], dataset_name: str):
    csv_path = "test_results.csv"
    file_exists = os.path.exists(csv_path)
    # [v8.3.2] Harmonized Comprehensive Metric Set
    headers = [
        "Timestamp", "Dataset", "Codec", "Images", "Orig_MB", "PNM_MB", "Comp_MB",
        "Saved_PNG%", "Saved_PNM%", "BPP", "Ratio_Mean", "Ratio_Med", 
        "Avg_Enc_ms", "Avg_Dec_ms", "Total_Enc_s", "Total_Dec_s",
        "Enc_TP", "Dec_TP", "Core_E_TP", "Core_D_TP",
        "Wins_S", "Wins_E", "Wins_D", "MSE"
    ]
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for s in stats_list:
            if not s: continue
            writer.writerow([
                now, dataset_name, s["name"], s.get("count", 0),
                f"{s['orig_mb']:.2f}", f"{s['pnm_mb']:.2f}", f"{s['comp_mb']:.2f}",
                f"{s['saved_pct']:.2f}", f"{s['saved_pnm_pct']:.2f}", f"{s['bpp']:.6f}",
                f"{s['mean_ratio']:.2f}", f"{s['median_ratio']:.2f}",
                f"{s.get('avg_e_ms', 0):.1f}", f"{s.get('avg_d_ms', 0):.1f}",
                f"{s.get('enc_wall_s', 0):.2f}", f"{s.get('dec_wall_s', 0):.2f}",
                f"{s['sys_tp'][0]:.2f}", f"{s['sys_tp'][1]:.2f}",
                f"{s['core_tp'][0]:.2f}", f"{s['core_tp'][1]:.2f}",
                s.get("wins_s", 0), s.get("wins_e", 0), s.get("wins_d", 0),
                f"{s['mse']:.10f}"
            ])

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SPX Comparative Benchmark Suite")
    parser.add_argument("codec", choices=["spx", "webp", "jxl", "bench"], help="Sub-command: webp, spx, jxl, or bench")
    parser.add_argument("path", nargs='?', help="Dataset path or alias (clic, gold, kodak, etc.)")
    parser.add_argument("--workers", "-w", type=int, default=os.cpu_count())
    parser.add_argument("--num_tests", "-n", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--bitplane", action="store_true", help="Force Bitplane engine")
    parser.add_argument("--reclassify", action="store_true", help="Copy files into easy/hard/hell categories")
    parser.add_argument("--build", nargs=4, metavar=('TARGET', 'E', 'H', 'HELL'), help="Construct dataset from categories")
    args = parser.parse_args()
    
    mapping = {
        "clic": "./data/CLIC",
        "gold": "./data/DIV2K_Gold",
        "div2k_gold": "./data/DIV2K_Gold",
        "val": "./data/DIV2K_Val",
        "validate": "./data/DIV2K_Val",
        "div2k_val": "./data/DIV2K_Val",
        "kodak": "./data/Kodak",
        "trgb": "./data/Tecnick_RGB",
        "tgray": "./data/Tecnick_Gray",
        "ici_rgb": "./data/Standard_RGB",
        "ici": "./data/Standard_RGB",
        "ici_gray": "./data/Standard_Gray",
        "train": "./data/DIV2K_Train",
        "div2k_train": "./data/DIV2K_Train",
        "full": "./data/DIV2K_Train",
        "easy": "./data/DIV2K_Easy",
        "div2k_easy": "./data/DIV2K_Easy",
        "hard": "./data/DIV2K_Hard",
        "div2k_hard": "./data/DIV2K_Hard",
        "hell": "./data/DIV2K_Hell",
        "div2k_hell": "./data/DIV2K_Hell"
    }

    # --- Build Priority Logic ---
    if args.build:
        target_alias, n_e, n_h, n_hell = args.build
        target_dir = mapping.get(target_alias.lower(), target_alias)
        n_e, n_h, n_hell = int(n_e), int(n_h), int(n_hell)
        
        print(f"[*] Building dataset {target_dir} (Easy:{n_e}, Hard:{n_h}, Hell:{n_hell})...")
        
        # Cleanup target
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            time.sleep(0.1)
        os.makedirs(target_dir, exist_ok=True)
        
        # Source folders
        src_base = "./data"
        categories = {
            "DIV2K_Easy": n_e,
            "DIV2K_Hard": n_h,
            "DIV2K_Hell": n_hell
        }
        
        total_copied = 0
        for cat_name, count in categories.items():
            src_dir = os.path.join(src_base, cat_name)
            if not os.path.exists(src_dir):
                print(f"  [!] Source {src_dir} missing. Skipping."); continue
            
            all_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.bmp', '.ppm', '.pgm', '.pnm'))]
            if len(all_files) < count:
                print(f"  [!] Warning: {src_dir} only has {len(all_files)} files (requested {count}). Using all.")
                selected = all_files
            else:
                selected = random.sample(all_files, count)
            
            for f in selected:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(target_dir, f))
                total_copied += 1
        
        print(f"Done. {total_copied} images assembled into {target_dir}.")
        if not args.path: return # Exit if only building

    if not args.path:
        print("[!] Error: Path is required for benchmarking."); return
    target_path = mapping.get(args.path.lower(), args.path)
    if not os.path.exists(target_path):
        print(f"[!] Error: Path {target_path} not found."); return

    if os.path.isdir(target_path):
        files = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.lower().endswith(('.png', '.bmp', '.ppm', '.pgm', '.pnm'))]
        files.sort()
        files = files[args.offset:args.offset+args.num_tests] if args.num_tests else files[args.offset:]
    else:
        files = [target_path]

    print("=" * 80)
    print(f"  SPX Unified Benchmark Hub (Target: {args.path.upper()} | Cores: {args.workers})")
    print("=" * 80)

    all_stats = []
    task_results = [] # List of (stats, {filename: res})
    
    queue = []
    if args.codec == "spx": queue = [("SPX", spx_worker)]
    elif args.codec == "webp": queue = [("SPX-Base", spx_worker), ("WebP(M6)", webp_worker)]
    elif args.codec == "jxl": queue = [("SPX-Base", spx_worker), ("JXL(E7)", jxl_worker)]
    elif args.codec == "bench": queue = [("SPX", spx_worker), ("WebP(M6)", webp_worker), ("JXL(E7)", jxl_worker)]

    try:
        for name, worker in queue:
            # [v8.3.2] Pass None for use_bitplane if not explicitly forced, to enable auto-selection
            bp_flag = True if args.bitplane else None
            stats, res_map = run_codec_benchmark(name, worker, files, args.workers, use_bitplane=bp_flag)
            # Ensure stats is at least a dict with the codec name if failure occurred
            if not stats:
                stats = {"name": name, "count": 0, "success": False}
            all_stats.append(stats)
            task_results.append(res_map if res_map else {})
    except (KeyboardInterrupt, Exception):
        sys.exit(1)

    # --- Win Counting Logic ---
    if len(task_results) > 1:
        fnames = set()
        for res_map in task_results:
            fnames.update(res_map.keys())
        fnames = sorted(list(fnames))
        
        wins = [ {"s":0, "e":0, "d":0} for _ in range(len(task_results)) ]
        
        for fn in fnames:
            # Gather metrics for this file across all codecs
            sizes, e_times, d_times = [], [], []
            skip = False
            for res_map in task_results:
                if fn not in res_map: skip = True; break
                sizes.append(res_map[fn]["comp_bytes"])
                e_times.append(res_map[fn]["e_s"])
                d_times.append(res_map[fn]["d_s"])
            if skip: continue
            
            # Find winners (index with minimum value)
            wins[np.argmin(sizes)]["s"] += 1
            wins[np.argmin(e_times)]["e"] += 1
            wins[np.argmin(d_times)]["d"] += 1
        
        # Inject Wins into stats
        for i, s in enumerate(all_stats):
            w = wins[i]
            s["wins_s"] = w["s"]
            s["wins_e"] = w["e"]
            s["wins_d"] = w["d"]

    print_comparison_table(all_stats, args.path.upper())
    export_to_csv(all_stats, args.path.upper())

    # --- Reclassification Logic ---
    norm_target = os.path.abspath(target_path).replace("\\", "/")
    if args.reclassify:
        # Restriction check: only reclassify if we have a valid SPX result set
        spx_map = None
        for i, (name, _) in enumerate(queue):
            if "SPX" in name:
                spx_map = task_results[i]
                break
        
        if not spx_map:
            print("[!] Skipping Reclassification: No SPX results available.")
        elif "div2k_train" not in norm_target.lower():
             print("[!] Warning: Reclassification usually expects DIV2K_Train dataset. Proceeding anyway...")
             
        if spx_map:
            data_root = os.path.dirname(target_path)
            print(f"[*] Pre-cleaning and Reclassifying images into DIV2K_Easy/Hard/Hell in {data_root}...")
            
            cat_dirs = {
                "easy": os.path.join(data_root, "DIV2K_Easy"),
                "hard": os.path.join(data_root, "DIV2K_Hard"),
                "hell": os.path.join(data_root, "DIV2K_Hell")
            }
            
            # Cleanup and Create directories
            for cat_path in cat_dirs.values():
                if os.path.exists(cat_path):
                    try:
                        shutil.rmtree(cat_path)
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"  [!] Warning: Could not fully clean {cat_path}: {e}")
                os.makedirs(cat_path, exist_ok=True)
            
            copy_count = 0
            for filename, res in spx_map.items():
                if not res["success"]: continue
                ratio = (res["comp_bytes"] / res["orig_bytes"]) * 100
                
                if ratio < 85:
                    cat_key = "easy"
                elif ratio < 95:
                    cat_key = "hard"
                else:
                    cat_key = "hell"
                
                src = os.path.join(target_path, filename)
                dst = os.path.join(cat_dirs[cat_key], filename)
                try:
                    shutil.copy2(src, dst)
                    copy_count += 1
                except Exception as e:
                    print(f"  [!] Failed to copy {filename}: {e}")
                    
            print(f"Done. {copy_count} images reclassified into DIV2K_ folders.")

if __name__ == "__main__":
    main()
