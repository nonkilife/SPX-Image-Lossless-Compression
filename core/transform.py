"""
SPX v8.3.2 [RCT Architecture]
Module: transform
Role: Pillar 3 - Spatial Transforms.
Description: High-speed G-sub RCT (Green-Subtract) and reconstruction kernels.
Architecture: Pillar-based modular spatial decorrelation layer.
Engineering Rationale:
1. Channel Dominance: Green (G) is used as the predictive lead because it usually 
   contains the highest structural information and correlates strongly with RD/BD 
   in natural sRGB images.
2. Zero-Loss Restoration: All spatial operations use uint8 modular arithmetic (wraparound) 
   to avoid the memory and CPU overhead of signed 16-bit intermediate buffers.

Technical Flowchart:
```mermaid
graph TD
    RGB[Input RGB/RGBA] --> GSUB[G-sub RCT: R-G, B-G]
    GSUB --> HIST[Row-wise Intensity Histograms]
    HIST --> CH[Channels: G, RD, BD, A]
    
    REC_CH[Reconstructed: G, RD, BD, A] --> IGSUB[Inverse G-sub: R=RD+G, B=BD+G]
    IGSUB --> OUT[Bit-Perfect RGB/RGBA]
```
"""

__version__ = "8.3.2"

import numpy as np
import numpy.typing as npt
from numba import njit, prange, uint8
from typing import Tuple, Optional
from .predictor import (ZIGZAG_LUT, IZIGZAG_LUT, selected_predictor)


@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def extract_channels(rgb: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint32]]:
    """
    Green-Subtract Reversible Color Transform (G-Sub RCT) and Intensity Mapping.
    [v8.3.2] Thread-safe histogram reduction using row-level accumulation.
    """
    if rgb.ndim == 2:
        h, w = rgb.shape
        gr_map = rgb.copy()
        rd_map = np.empty((0, 0), dtype=np.uint8)
        bd_map = np.empty((0, 0), dtype=np.uint8)
        a_map = np.empty((0, 0), dtype=np.uint8)
        
        row_hists = np.zeros((h, 256), dtype=np.uint32)
        for i in prange(h):
            for j in range(w):
                row_hists[i, rgb[i, j]] += 1
        
        global_hists = np.zeros((3, 256), dtype=np.uint32)
        # Final reduction (Parallelized over symbols for efficiency)
        for val in prange(256):
            acc = np.uint32(0)
            for i in range(h):
                acc += row_hists[i, val]
            global_hists[0, val] = acc
            
        return gr_map, rd_map, bd_map, a_map, global_hists

    h, w, c = rgb.shape[0], rgb.shape[1], rgb.shape[2]
    gr_map = np.empty((h, w), dtype=np.uint8)
    rd_map = np.empty((h, w), dtype=np.uint8)
    bd_map = np.empty((h, w), dtype=np.uint8)
    a_map = np.empty((h, w), dtype=np.uint8) if c == 4 else np.empty((0, 0), dtype=np.uint8)
    
    # [v8.3.2] Row-wise histograms to avoid execute-time race conditions in prange
    row_hists = np.zeros((h, 3, 256), dtype=np.uint32)
    
    for i in prange(h):
        for j in range(w):
            pix = rgb[i, j]
            r, g, b = pix[0], pix[1], pix[2]
            gr_map[i, j] = g
            rd_v = np.uint8((int(r) - int(g)) & 0xFF)
            bd_v = np.uint8((int(b) - int(g)) & 0xFF)
            rd_map[i, j] = rd_v
            bd_map[i, j] = bd_v
            
            row_hists[i, 0, g] += 1
            row_hists[i, 1, rd_v] += 1
            row_hists[i, 2, bd_v] += 1
            if c == 4:
                a_map[i, j] = pix[3]
    
    # Final global reduction
    global_hists = np.zeros((3, 256), dtype=np.uint32)
    for ch in range(3):
        for val in prange(256):
            acc = np.uint32(0)
            for i in range(h):
                acc += row_hists[i, ch, val]
            global_hists[ch, val] = acc

    return gr_map, rd_map, bd_map, a_map, global_hists

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def restore_channels(gr_rec: npt.NDArray[np.uint8], rd_rec: npt.NDArray[np.uint8],
                     bd_rec: npt.NDArray[np.uint8], a_ch: npt.NDArray[np.uint8],
                     is_rgba: bool, is_grayscale: bool, apply_gsub: bool) -> npt.NDArray[np.uint8]:
    """ Inverse G-sub RCT transform (Parallelized). apply_gsub mirrors FLAG_COLOR_GSUB. """
    h, w = gr_rec.shape
    rgb = np.zeros((h, w, 4 if is_rgba else 3), dtype=np.uint8)
    
    for i in prange(h):
        for j in range(w):
            g = gr_rec[i, j]
            if is_grayscale:
                r_out, g_out, b_out = g, g, g
            elif apply_gsub:
                r_out = np.uint8((int(rd_rec[i, j]) + int(g)) & 0xFF)
                g_out = g
                b_out = np.uint8((int(bd_rec[i, j]) + int(g)) & 0xFF)
            else:
                r_out, g_out, b_out = rd_rec[i, j], g, bd_rec[i, j]
            
            rgb[i, j, 0] = r_out
            rgb[i, j, 1] = g_out
            rgb[i, j, 2] = b_out
            if is_rgba:
                rgb[i, j, 3] = a_ch[i, j]
                
    return rgb

@njit(parallel=True, cache=True)
def predict_2d_residuals(data_ch: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Applies MED prediction rowwise and encodes each residual as a ZigZag symbol via LUT.
    [v8.3.2] Used primarily for Alpha and non-sharded grayscale modes.
    """
    h, w = data_ch.shape
    res = np.empty((h, w), dtype=uint8)
    if h == 0 or w == 0: return res
    
    for i in prange(h):
        # Boundary pixel (j=0)
        b = data_ch[i-1, 0] if i > 0 else uint8(0)
        pred = selected_predictor(uint8(0), b, uint8(0))
        res[i, 0] = ZIGZAG_LUT[np.uint8((int(data_ch[i, 0]) - int(pred)) & 0xFF)]

        for j in range(1, w):
            a = data_ch[i, j-1]
            b = data_ch[i-1, j] if i > 0 else uint8(0)
            c = data_ch[i-1, j-1] if i > 0 else uint8(0)
            pred = selected_predictor(a, b, c)
            res[i, j] = ZIGZAG_LUT[np.uint8((int(data_ch[i, j]) - int(pred)) & 0xFF)]
    return res

@njit(fastmath=True, cache=True)
def reconstruct_2d_channels(h: int, w: int, res_ch: npt.NDArray[np.uint8], out: Optional[npt.NDArray[np.uint8]] = None) -> npt.NDArray[np.uint8]:
    """ Inverse MED reconstruction from a 2D ZigZag residual matrix using IZIGZAG_LUT. """
    rec = out if out is not None else np.zeros((h, w), dtype=uint8)
    if h == 0 or w == 0: return rec
    
    for i in range(h):
        # Boundary pixel (j=0)
        b = rec[i-1, 0] if i > 0 else uint8(0)
        pred = selected_predictor(uint8(0), b, uint8(0))
        rec[i, 0] = uint8((int(IZIGZAG_LUT[res_ch[i, 0]]) + int(pred)) & 0xFF)

        for j in range(1, w):
            a = rec[i, j-1]
            b = rec[i-1, j] if i > 0 else uint8(0)
            c = rec[i-1, j-1] if i > 0 else uint8(0)
            pred = selected_predictor(a, b, c)
            rec[i, j] = uint8((int(IZIGZAG_LUT[res_ch[i, j]]) + int(pred)) & 0xFF)
    return rec
