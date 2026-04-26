r"""
SPX v8.3.2 [RCT Architecture]
Module: transform
Role: Pillar 3 - Spatial Transforms.

Description: 
High-speed G-sub RCT (Green-Subtract) and reconstruction kernels. This module 
handles the conversion between RGB and decorrelated residual space.

Engineering Rationale:
1. Channel Dominance: Green (G) is used as the predictive lead because it usually 
   contains the highest structural information and correlates strongly with RD/BD 
   in natural sRGB images. By subtracting G from R and B, we significantly 
   reduce the energy (entropy) of the chroma channels.
2. Zero-Loss Restoration: All spatial operations use uint8 modular arithmetic (wraparound). 
   Because the transform is reversible ($R = RD + G \pmod{256}$), we avoid the need 
   for signed 16-bit intermediate buffers, saving memory and cache bandwidth.
3. 2D Reconstruction: Residuals are stored in 1D shard bitstreams. This module 
   rebuilds the 2D spatial context using the MED predictor (Pillar 2) to 
   reconstruct the original pixel values.

Technical Flowchart:
```mermaid
graph TD
    REC_CH[Reconstructed: G, RD, BD, A] --> IGSUB[Inverse G-sub: R=RD+G, B=BD+G]
    IGSUB --> OUT[Bit-Perfect RGB/RGBA]
```
"""

__version__ = "8.3.2"

__all__ = ['restore_channels', 'reconstruct_2d_channels']

import numpy as np
import numpy.typing as npt
from typing import Optional
import spx_rans as _rs


def restore_channels(gr_rec: npt.NDArray[np.uint8], rd_rec: npt.NDArray[np.uint8],
                     bd_rec: npt.NDArray[np.uint8], a_ch: npt.NDArray[np.uint8],
                     is_rgba: bool, is_grayscale: bool, apply_gsub: bool) -> npt.NDArray[np.uint8]:
    """ 
    Inverse G-sub RCT transform (Rust backend, parallel). 
    Reverses the color decorrelation by adding the Green channel back to 
    the Red-diff and Blue-diff channels.
    """
    return _rs.restore_channels(
        np.ascontiguousarray(gr_rec, dtype=np.uint8),
        np.ascontiguousarray(rd_rec, dtype=np.uint8),
        np.ascontiguousarray(bd_rec, dtype=np.uint8),
        np.ascontiguousarray(a_ch,   dtype=np.uint8),
        bool(is_rgba), bool(is_grayscale), bool(apply_gsub)
    )

def reconstruct_2d_channels(h: int, w: int, res_ch: npt.NDArray[np.uint8],
                             out: Optional[npt.NDArray[np.uint8]] = None) -> npt.NDArray[np.uint8]:
    """ 
    Inverse MED reconstruction from a 2D ZigZag residual matrix (Rust backend). 
    Iteratively reconstructs pixels using the Edge-Tuned MED predictor.
    """
    result = _rs.reconstruct_2d_channels(h, w, np.ascontiguousarray(res_ch, dtype=np.uint8))
    if out is not None:
        out[:] = result
        return out
    return result
