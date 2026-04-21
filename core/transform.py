"""
SPX [RCT Architecture]
Module: transform
Role: Pillar 3 - Spatial Transforms.
Description: High-level image transformations and reconstruction kernels.
Architecture: RGB/RGBA Channel management and G-sub Recursive Color Transform.
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

import numpy as np
import numpy.typing as npt
from numba import njit, prange, uint8
from typing import Tuple
from .common import (to_zigzag, from_zigzag, selected_predictor)

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def extract_channels(rgb: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint32]]:
    """
    Green-Subtract Reversible Color Transform (G-Sub RCT) and Intensity Mapping.
    
    Operation:
    Extracts the G channel directly, then subtracts G from R and B.
    By doing so, the luminance correlation (which heavily permeates G, R, B simultaneously) 
    is largely eliminated from R and B, making these newly created RD and BD color-difference 
    channels highly compressible. Also outputs per-row intensity histograms for profile logic.
    """
    h, w, c = rgb.shape[0], rgb.shape[1], rgb.shape[2]
    gr_map = np.empty((h, w), dtype=np.uint8)
    rd_map = np.empty((h, w), dtype=np.uint8)
    bd_map = np.empty((h, w), dtype=np.uint8)
    a_map = np.empty((h, w), dtype=np.uint8) if c == 4 else np.empty((0, 0), dtype=np.uint8)
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


    global_hists = row_hists.sum(axis=0)
    return gr_map, rd_map, bd_map, a_map, global_hists

@njit(parallel=True, fastmath=True, error_model='numpy', cache=True)
def restore_channels(gr_rec: npt.NDArray[np.uint8], rd_rec: npt.NDArray[np.uint8],
                     bd_rec: npt.NDArray[np.uint8], a_ch: npt.NDArray[np.uint8],
                     is_rgba: bool, is_grayscale: bool, apply_gsub: bool) -> npt.NDArray[np.uint8]:
    """ Inverse G-sub RCT transform (Parallelized). apply_gsub mirrors FLAG_COLOR_GSUB. """
    h, w = gr_rec.shape
    rgb = np.zeros((h, w, 4 if is_rgba else 3), dtype=np.uint8)
    
    if is_grayscale:
        for i in prange(h):
            for j in range(w):
                g = gr_rec[i, j]
                rgb[i, j, 0], rgb[i, j, 1], rgb[i, j, 2] = g, g, g
                if is_rgba: rgb[i, j, 3] = a_ch[i, j]
    elif apply_gsub:
        for i in prange(h):
            for j in range(w):
                g = gr_rec[i, j]
                r = np.uint8((int(rd_rec[i, j]) + int(g)) & 0xFF)
                b = np.uint8((int(bd_rec[i, j]) + int(g)) & 0xFF)
                rgb[i, j, 0], rgb[i, j, 1], rgb[i, j, 2] = r, g, b
                if is_rgba: rgb[i, j, 3] = a_ch[i, j]
    else:
        for i in prange(h):
            for j in range(w):
                g = gr_rec[i, j]
                rgb[i, j, 0], rgb[i, j, 1], rgb[i, j, 2] = rd_rec[i, j], g, bd_rec[i, j]
                if is_rgba: rgb[i, j, 3] = a_ch[i, j]
    return rgb

@njit(error_model='numpy', cache=True)
def decode_alpha_channel(h: int, w: int, res_ch: npt.NDArray[np.uint8], rec_ch: npt.NDArray[np.uint8]) -> None:
    """ Predictive restoration for Alpha channel. """
    for i in range(h):
        row_rec, res_row = rec_ch[i], res_ch[i]
        row_trec = rec_ch[i-1] if i > 0 else rec_ch[0]
        a_val = np.uint8(0)
        c_val = np.uint8(0)  # reset each row; holds upper-left diagonal for next pixel
        for j in range(w):
            b_val = row_trec[j] if i > 0 else np.uint8(0)
            p = selected_predictor(a_val, b_val, c_val)
            val = np.uint8((from_zigzag(res_row[j]) + int(p)) & 0xFF)
            row_rec[j] = val
            a_val = val
            c_val = row_trec[j] if i > 0 else np.uint8(0)

@njit(parallel=True, cache=True)
def predict_2d_residuals(data_ch: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Applies MED prediction rowwise and encodes each residual as a ZigZag symbol.
    Returns a 2D residual matrix with the same shape as the input channel.
    """
    h, w = data_ch.shape
    res = np.empty((h, w), dtype=uint8)
    if h == 0 or w == 0:
        return res
    for i in prange(h):
        # Boundary pixel (j=0): no left or upper-left neighbor
        b = data_ch[i-1, 0] if i > 0 else uint8(0)
        pred = selected_predictor(uint8(0), b, uint8(0))
        res[i, 0] = to_zigzag(int(data_ch[i, 0]) - int(pred))

        for j in range(1, w):
            a = data_ch[i, j-1]
            b = data_ch[i-1, j] if i > 0 else uint8(0)
            c = data_ch[i-1, j-1] if i > 0 else uint8(0)
            pred = selected_predictor(a, b, c)
            res[i, j] = to_zigzag(int(data_ch[i, j]) - int(pred))
    return res

@njit(cache=True)
def reconstruct_2d_channels(h: int, w: int, res_ch: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """ Inverse MED reconstruction from a 2D ZigZag residual matrix. """
    rec = np.zeros((h, w), dtype=uint8)
    if h == 0 or w == 0:
        return rec
    for i in range(h):
        # Boundary pixel (j=0): no left or upper-left neighbor
        b = rec[i-1, 0] if i > 0 else uint8(0)
        pred = selected_predictor(uint8(0), b, uint8(0))
        rec[i, 0] = uint8((from_zigzag(res_ch[i, 0]) + int(pred)) & 0xFF)

        for j in range(1, w):
            a = rec[i, j-1]
            b = rec[i-1, j] if i > 0 else uint8(0)
            c = rec[i-1, j-1] if i > 0 else uint8(0)
            pred = selected_predictor(a, b, c)
            rec[i, j] = uint8((from_zigzag(res_ch[i, j]) + int(pred)) & 0xFF)
    return rec

@njit(parallel=True, fastmath=True, cache=True)
def calculate_aad_estimate(data_ch: npt.NDArray[np.uint8]) -> float:
    """ Fast AAD (Average Absolute Deviation) estimation using MED residuals. """
    h, w = data_ch.shape
    if h == 0 or w == 0:
        return 0.0
    total_abs_err = 0.0
    for i in prange(h):
        row_sum = 0.0
        for j in range(w):
            a = data_ch[i, j-1] if j > 0 else uint8(0)
            b = data_ch[i-1, j] if i > 0 else uint8(0)
            c = data_ch[i-1, j-1] if (i > 0 and j > 0) else uint8(0)
            pred = selected_predictor(a, b, c)
            row_sum += abs(float(np.int32(data_ch[i, j]) - np.int32(pred)))
        total_abs_err += row_sum
    return total_abs_err / (h * w)
