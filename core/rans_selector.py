"""
SPX v8.3.2 PDF Template Selector (rans_selector)
Module: rans_selector
Role: Pillar 0 - Meta-Decision Engine.
Description: Thin Python shim over the spx_rans Rust native backend.

All decision logic (cross-entropy, PDF normalization, template evaluation)
is implemented in native/src/rans_core.rs (decide_shard_mode).

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

Decision Logic (Cross-Entropy vs Penalty):
The selector evaluates the theoretical bit-cost (cross-entropy) of encoding the data using the perfectly
fitted Custom PDF, versus the "best-fit" Static Template. Since the Custom PDF is perfectly fitted, it will
ALWAYS yield the lowest mathematical bit-cost. However, it applies a 'penalty' to the Custom PDF's cost to
simulate the physical file size required to save the table into the header. If the penalty makes the Custom
PDF more expensive than the "slightly ill-fitting but free" Static Template, the Template is chosen.

Conceptual Architecture:
-------------------------
- Sub-mode decision (Dense vs Sparse) is finalized during serialization in rans.py.
- Mode 3 is prioritized for mono-symbol (zero-entropy) payloads.
"""

__version__ = "8.3.2"

import os
import numpy as np
import numpy.typing as npt

import spx_rans as _rs
from .common import get_empirical_templates


def decide_shard_mode(
    counts: npt.NDArray[np.uint64],
    width: int,
    header_penalty_bits: float = 120.0,
) -> tuple[int, npt.NDArray[np.uint64]]:
    """
    Heuristic decision engine delegating to the Rust native backend.
    Default header penalty (120 bits) represents a 15-byte serialization cost.
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
