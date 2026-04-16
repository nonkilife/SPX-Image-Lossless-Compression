"""
ZPNG-CSDE v7.3 [Dynamic Shard Routing]
Module: zpng_shard_lut
Role: Pillar 3.6 - Dynamic Mode Dispatch
Description: Static Look-up table definitions and Causal Decision Tree for zero-overhead shard routing.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, uint8

# --- 1. Dynamic Routing Constants ---

# Predictor Engine IDs
PRED_MED: int       = 0
PRED_GAP: int       = 1
PRED_GRADIENT: int  = 2
PRED_LEFT: int      = 3

# Entropy Coder Types
CODER_BIT_RANS: int = 0
CODER_RGB_RANS: int = 1

def _build_shard_config_lut() -> npt.NDArray[np.uint8]:
    """
    Constructs the 128-entry dynamic routing configuration table.
    Format per entry: [predictor_id, rans_type, rans_mode]
    """
    lut: npt.NDArray[np.uint8] = np.zeros((128, 3), dtype=np.uint8)
    
    # Phase 1: Initialize baseline defaults -> MED + Bit rANS.
    for i in range(128):
        lut[i, 0] = PRED_MED
        lut[i, 1] = CODER_BIT_RANS
        lut[i, 2] = 0  # Default mode 0
        
    return lut

# Global Immutable Instance
SHARD_CONFIG_LUT: npt.NDArray[np.uint8] = _build_shard_config_lut()

@njit(inline='always', cache=True)
def get_lut_index_from_context(ctx_id: uint8) -> uint8:
    """
    Zero-overhead Causal Decision Tree Kernel.
    Evaluates context id and historical boundaries to assign a routing LUT index.
    """
    # Phase 1: Direct 1:1 mapping placeholder.
    # Will be expanded with spatial causal features in Phase 4.
    return uint8(int(ctx_id) & 127)
