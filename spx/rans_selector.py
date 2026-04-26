"""
SPX v8.3.2 PDF Template Selector (rans_selector)
Module: rans_selector
Role: Pillar 0 - Meta-Decision Engine.

Description: 
Thin Python shim over the spx_rans Rust native backend. This module implements 
the logic for selecting the optimal entropy coding model for each shard.

Technical Flowchart:
```mermaid
graph TD
    Input[Shard Byte-Stream] --> Build[Build Custom PDF]
    Build --> Cross[Calculate Cross-Entropy for Templates 4-33]
    Cross --> Penalty{Custom PDF + Penalty < Best Template?}
    Penalty -->|Yes| Mode0[Mode 0: Custom Dynamic PDF]
    Penalty -->|No| ModeT[Mode 4-33: Hardcoded Template]

    Mode0 --> SubMode{Sub-Mode Decision}
    SubMode -->|Dense| SM0[Sub-Mode 0: Full Array]
    SubMode -->|Sparse| SM1[Sub-Mode 1: Index-Value Pairs]
    ModeT --> Header[Header: 1-byte Mode Marker]
```

Architecture & Engineering Rationale:
1. Decision Logic (Cross-Entropy vs Penalty): The selector evaluates the 
   theoretical bit-cost (cross-entropy) of encoding the data using the perfectly 
   fitted Custom PDF, versus the "best-fit" Static Template. Since the Custom PDF 
   is perfectly fitted, it will ALWAYS yield the lowest mathematical bit-cost. 
   However, it applies a 'penalty' to the Custom PDF's cost to simulate the 
   physical file size required to save the table into the header. If the penalty 
   makes the Custom PDF more expensive than the "slightly ill-fitting but free" 
   Static Template, the Template is chosen.
2. Mode 3 Prioritization: Mode 3 (Zero-Entropy) is automatically selected for 
   shards containing only a single symbol (usually zero). This eliminates the 
   need for any rANS state transitions or bitstream payloads for that shard.
3. Sub-Mode Decision: For Mode 0 (Custom PDF), the engine chooses between 
   'Dense' (full array) and 'Sparse' (index-value pairs) based on which 
   representation minimizes the header overhead.

Mode Design & Composition:
-------------------------
- Mode 0: Custom Dynamic Header (Variable overhead)
    - Composition: User-defined 12-bit PDF array. Utilizes an internal sub-selector:
        - Sub-Mode 0: Dense (Full ZigZag width array).
        - Sub-Mode 1: Sparse (Index-Value pairs for high-energy outliers).
    - Physical Shape: [0x00][Sub-Mode:1][Payload:10-530 bytes].
    - Design Goal: Mathematical optimality for heavy-tail or irregular data distributions.
- Mode 3: Zero-Entropy (Empty/Flat)
    - Composition: Virtual delta distribution [4096, 0, 0...].
    - Physical Shape: [0x03] (Single byte).
    - Design Goal: Perfect efficiency for zero-residual shards in flat regions.
- Mode 4-33: Hybrid Empirical Templates (Zero overhead)
    - Composition: 30 pre-computed curves (10 Hybrid Centroids x 3 Scales [0.5, 1.0, 1.5]).
    - Physical Shape: [ModeID:1] (Single byte).
    - Design Goal: Zero-tax modeling for standard natural image gradients.

Conceptual Architecture:
-------------------------
- Sub-mode decision (Dense vs Sparse) is finalized during serialization in rans.py.
- Mode 3 is prioritized for mono-symbol (zero-entropy) payloads.

All decision logic is implemented in native/src/rans_core.rs (decide_shard_mode).
"""

__version__ = "8.3.2"

import os
import numpy as np
import numpy.typing as npt
from typing import Tuple

import spx_rans as _rs
from .common import get_empirical_templates

__all__ = ['decide_shard_mode']


def decide_shard_mode(
    counts: npt.NDArray[np.uint64],
    width: int,
    header_penalty_bits: float = 120.0,
) -> Tuple[int, npt.NDArray[np.uint64]]:
    """
    Heuristic decision engine delegating to the Rust native backend.
    
    Args:
        counts: Histogram of residuals in the shard.
        width: ZigZag spread of the residuals.
        header_penalty_bits: Virtual bit-cost of a Custom PDF header. 
                             Default (120 bits) ~ 15 bytes.
    
    Returns:
        (mode_id, normalized_pdf)
    """
    templates = get_empirical_templates()
    disable = os.environ.get("SPX_DISABLE_TEMPLATES") == "1"
    mode, pdf = _rs.decide_shard_mode(
        np.ascontiguousarray(counts, dtype=np.uint64),
        int(width),
        float(header_penalty_bits),
        np.ascontiguousarray(templates, dtype=np.uint64),
        bool(disable),
    )
    return int(mode), pdf
