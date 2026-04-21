"""
SPX PDF Template Selector (rans_selector)
Module: rans_selector
Role: Pillar 0 - Meta-Decision Engine.
Description: Heuristic selector for optimal entropy modeling per shard.

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
"""

import numpy as np
import numpy.typing as npt
from numba import njit, uint8, uint16, uint32, uint64
import os

from .common import EMPIRICAL_TEMPLATES

@njit(fastmath=True, cache=True)
def calculate_cross_entropy(counts: npt.NDArray[np.uint64], pdf: npt.NDArray[np.uint16]) -> float:
    """ Calculates BPP representation: Sum(counts[i] * -log2(pdf[i]/4096)) """
    entropy = 0.0
    # Pre-scale pdf values to avoid per-pixel division; 4096 is the sum(pdf)
    # entropy = counts * (log2(4096) - log2(pdf))
    # log2(4096) = 12.0
    for i in range(256):
        if counts[i] > 0:
            entropy += float(counts[i]) * (12.0 - np.log2(float(pdf[i])))
    return entropy

@njit(fastmath=True, cache=True)
def build_pdf_from_counts(counts: npt.NDArray[np.uint64], width: int) -> npt.NDArray[np.uint16]:
    """
    Precision Alphabet-Tightening PDF Builder.
    Generates a normalized 12-bit (4096 total) PDF only for symbols within [0, width-1].
    Symbols outside this range are strictly 0 probability, eliminating the "Survival Tax".
    """
    total = np.sum(counts)
    active_limit = min(256, int(width))
    if total == 0:
        pdf = np.zeros(256, dtype=np.uint16)
        pdf[:active_limit] = uint16(max(1, 4096 // active_limit))
        # Fix total sum to 4096
        rem = 4096 - np.sum(pdf)
        pdf[0] += uint16(rem)
        return pdf
        
    pdf = np.zeros(256, dtype=np.uint16)
    # 1+2. Rational normalization with inline min-prob enforcement (single pass)
    current_sum = 0
    for i in range(active_limit):
        v = uint16(round((float(counts[i]) / float(total)) * 4096.0))
        if counts[i] > 0 and v == 0:
            v = uint16(1)
        pdf[i] = v
        current_sum += int(v)

    # 3. Correct total mass to exactly 4096 using the Peak Distribution strategy
    # Note: Symbols [active_limit : 256] remain 0.
    diff = int(4096) - int(current_sum)

    # Use argsort to find largest symbols for distribution of diff
    # We find the peak ONLY within the active range
    peak_idx = 0
    max_val = uint16(0)
    for i in range(active_limit):
        if pdf[i] > max_val:
            max_val = pdf[i]
            peak_idx = i
            
    new_peak = int(pdf[peak_idx]) + diff
    if new_peak >= 1:
        pdf[peak_idx] = uint16(new_peak)
    else:
        # Emergency redistribution if peak cannot absorb
        pdf[peak_idx] = uint16(1)
        remaining = 1 - new_peak
        for i in range(active_limit):
            if remaining <= 0: break
            if i != peak_idx and int(pdf[i]) > 1:
                cut = min(int(pdf[i]) - 1, remaining)
                pdf[i] = uint16(int(pdf[i]) - cut)
                remaining -= cut

    return pdf

# Non-njit helper to check env
def is_templates_disabled() -> bool:
    return os.environ.get("ZPNG_DISABLE_TEMPLATES") == "1"

@njit(fastmath=True, cache=True)
def _decide_shard_mode_core(counts: npt.NDArray[np.uint64], width: int, 
                            header_penalty_bits: float, 
                            templates: npt.NDArray[np.uint64],
                            disable_templates: bool) -> tuple[uint8, npt.NDArray[np.uint64]]:
    
    # 1. Build Custom Dynamic PDF
    dense_pdf = build_pdf_from_counts(counts, width)

    if disable_templates:
        return uint8(0), dense_pdf.astype(np.uint64)

    # 2. Empirical Templates (Mode 4-33) Evaluation
    best_emp_mode = uint8(0)
    min_emp_bits = 1e18
    
    # Pixel-Adaptive Dynamic Penalty: adjusts relative to shard size.
    # For small images/shards, the physical byte size of the custom table represents a massive % of the payload,
    # thus the penalty (discouraging Mode 0) becomes exponentially higher.
    n_pixels = uint32(0)
    for i in range(256): n_pixels += uint32(counts[i])
    penalty = header_penalty_bits * (4096.0 / max(float(n_pixels), 1.0))
    
    num_templates = len(templates)
    for tid in range(num_templates):
        tpl = templates[tid]
        tpl_bits = calculate_cross_entropy(counts, tpl.astype(uint16))
        if tpl_bits < min_emp_bits:
            min_emp_bits = tpl_bits
            best_emp_mode = uint8(4 + tid)
            
    # Calculate the exact entropy of the custom table, and add the "Header Tax" penalty.
    dense_bits = calculate_cross_entropy(counts, dense_pdf.astype(uint16))
    dense_total_bits = dense_bits + penalty
    
    if min_emp_bits < dense_total_bits:
        return best_emp_mode, templates[int(best_emp_mode) - 4]
    else:
        return uint8(0), dense_pdf.astype(np.uint64)

def decide_shard_mode(counts: npt.NDArray[np.uint64], width: int, header_penalty_bits: float = 120.0) -> tuple[uint8, npt.NDArray[np.uint64]]:
    """ 
    Heuristic decision engine with Audit Switch.
    Default header penalty (120 bits) represents a 15-byte serialization cost.
    """
    disable = is_templates_disabled()
    # Pass templates array for Njit compatibility
    return _decide_shard_mode_core(counts, width, header_penalty_bits, EMPIRICAL_TEMPLATES, disable)
