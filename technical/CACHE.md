# SPX Cache Consumption Audit (v8.3.2)

This document outlines the memory footprint and CPU cache residency of the SPX codec's core components. Understanding this hierarchy is critical for optimizing throughput and avoiding cache thrashing.

## 1. Memory Hierarchy Overview

The SPX architecture is designed to keep hot-path data within the L1 and L2 caches to minimize memory latency.

| Cache Level | Primary Residents | Footprint (Est.) | Status |
| :--- | :--- | :--- | :--- |
| **L1 (32-64 KB)** | rANS Models, Bitplane Contexts (2.6K), ZigZag LUTs | ~16 KB | **Excellent** |
| **L2 (512 KB - 1 MB)** | Spatial LUT (`s_lut`), Profiling Histograms, Shard Offsets | ~600 KB | **Optimized** |
| **L3 (16 MB+)** | Image Channel Buffers (RGBA), Bitstream Payloads | 4 - 32 MB | **Stable** |

> [!NOTE]
> **Rust-Native Backend Impact**: The migration to a Rust-native backend in v8.3.2 eliminates Python VM overhead and JIT compilation spikes during the hot path. All memory operations within the core kernels are performed via direct pointer arithmetic on contiguous buffers, reducing memory-bound stalls.

---

## 2. Component Breakdown

### A. Context Derivation (Pillar 4 Foundation)
The most frequently accessed tables during the sharding process.
*   **`s_lut` (Spatial Feature LUT)**: $511 \times 511 \times 1$ byte = **256 KB**
*   **`i_lut` (Intensity Segment LUT)**: $256 \times 1$ byte = **256 Bytes**
*   **`d_lut` (Dispatch/Shard ID LUT)**: $256 \times 4$ bytes = **1 KB**
*   **Total Footprint**: **~257 KB**
*   **Residency**: Permanent L2 residency during `shard_pass`.

### B. Prediction & Mapping (Pillar 2)
Static lookup tables used for residual normalization.
*   **ZigZag LUTs** (`ZIGZAG`, `IZIGZAG`, `BICC`): $256 \times 3$ = **768 Bytes**
*   **Residency**: L1 Cache.

### C. Entropy Coding (rANS)
Models are swapped per shard but fit within L1.
*   **Standard rANS**: `slot_lookup` (4096) + `symbol_freqs` = **8 KB**
*   **Bitplane rANS**: 2,688-way spatial context model ($42 \times 64$) = **~10 KB**
*   **Residency**: L1 Cache (Core execution loop).

### D. Intermediate Working Buffers
Allocated dynamically during the encoding profiling pass.
*   **`local_hists`**: $3 \text{ channels} \times 42 \text{ shards} \times 256 \text{ symbols} \times 4 \text{ bytes}$ = **126 KB**
*   **`row_ptrs` / `row_offsets`**: 512 rows $\times$ 3 ch $\times$ 42 shards $\times$ 4 bytes = **~252 KB**
*   **Residency**: L2 Cache.

---

## 3. Optimization Rationale

### The "Indirection vs. Footprint" Trade-off
Current performance is limited by the number of lookup steps (3 steps for context derivation). 
*   **Proposed "Multi-Plane" s_lut**: Merging `s_lut` and `d_lut` into a $(4, 511, 511)$ array.
    *   **New Footprint**: **~1.02 MB**
    *   **Impact**: May slightly stress L2 on older CPUs but eliminates one level of indirection, which is typically a net gain for superscalar execution.

### Negative Optimization: 3D Predictor LUT
A $256^3$ 3D LUT would consume **16.7 MB**.
*   **Warning**: This would displace the primary image buffers from L3 cache, leading to severe cache pollution and potentially *slower* execution despite fewer arithmetic instructions.
*   **Decision**: Rejected in favor of the **Branchless Edge-Tuned MED** (v8.3.2), which achieves ~30% gain via arithmetic clamping without any additional memory overhead.

---

## 4. Performance Constraints

With the removal of JIT latency and branch-heavy logic, the system has transitioned from being **Instruction-Bound** to being **Dependency-Bound**.

*   **Instruction-Level Parallelism (ILP)**: 4-way interleaved rANS mitigates the serial dependency of the rANS state update, but throughput remains limited by the CPU's ability to resolve the recurrence within the entropy core.
*   **Memory Safety**: Rust-native execution ensures strict boundary checks with minimal performance penalty, providing a stable execution environment compared to dynamic Python/Numba environments.
