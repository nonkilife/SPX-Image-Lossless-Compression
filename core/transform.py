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
from typing import Optional
import spx_rans as _rs


def restore_channels(gr_rec: npt.NDArray[np.uint8], rd_rec: npt.NDArray[np.uint8],
                     bd_rec: npt.NDArray[np.uint8], a_ch: npt.NDArray[np.uint8],
                     is_rgba: bool, is_grayscale: bool, apply_gsub: bool) -> npt.NDArray[np.uint8]:
    """ Inverse G-sub RCT transform (Rust backend, parallel). apply_gsub mirrors FLAG_COLOR_GSUB. """
    return _rs.restore_channels(
        np.ascontiguousarray(gr_rec, dtype=np.uint8),
        np.ascontiguousarray(rd_rec, dtype=np.uint8),
        np.ascontiguousarray(bd_rec, dtype=np.uint8),
        np.ascontiguousarray(a_ch,   dtype=np.uint8),
        bool(is_rgba), bool(is_grayscale), bool(apply_gsub)
    )

def reconstruct_2d_channels(h: int, w: int, res_ch: npt.NDArray[np.uint8],
                             out: Optional[npt.NDArray[np.uint8]] = None) -> npt.NDArray[np.uint8]:
    """ Inverse MED reconstruction from a 2D ZigZag residual matrix (Rust backend). """
    result = _rs.reconstruct_2d_channels(h, w, np.ascontiguousarray(res_ch, dtype=np.uint8))
    if out is not None:
        out[:] = result
        return out
    return result
