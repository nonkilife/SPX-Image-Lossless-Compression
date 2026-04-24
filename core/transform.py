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

@njit(fastmath=True, cache=True)
def _reconstruct_2d_inplace(h: int, w: int, res_ch: npt.NDArray[np.uint8],
                              rec: npt.NDArray[np.uint8]) -> None:
    """ JIT kernel: writes inverse-MED result into pre-allocated rec array. """
    for i in range(h):
        b = rec[i-1, 0] if i > 0 else uint8(0)
        pred = selected_predictor(uint8(0), b, uint8(0))
        rec[i, 0] = uint8((int(IZIGZAG_LUT[res_ch[i, 0]]) + int(pred)) & 0xFF)
        for j in range(1, w):
            a = rec[i, j-1]
            b = rec[i-1, j] if i > 0 else uint8(0)
            c = rec[i-1, j-1] if i > 0 else uint8(0)
            pred = selected_predictor(a, b, c)
            rec[i, j] = uint8((int(IZIGZAG_LUT[res_ch[i, j]]) + int(pred)) & 0xFF)

def reconstruct_2d_channels(h: int, w: int, res_ch: npt.NDArray[np.uint8],
                             out: npt.NDArray[np.uint8] = None) -> npt.NDArray[np.uint8]:
    """ Inverse MED reconstruction from a 2D ZigZag residual matrix using IZIGZAG_LUT. """
    rec = out if out is not None else np.zeros((h, w), dtype=np.uint8)
    if h > 0 and w > 0:
        _reconstruct_2d_inplace(h, w, res_ch, rec)
    return rec
