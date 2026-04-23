# SPX Native Backend (Rust) Conversion Learnings

This document records the technical challenges, architectural pitfalls, and performance regressions encountered during the initial phase of transitioning the SPX codec from Numba to a Rust native backend.

## 1. Threading Over-subscription (Critical Performance Warning)
**Finding:** Integrating a multi-threaded Rust library (via `rayon`) into a multi-process Python environment (via `multiprocessing`) leads to catastrophic performance degradation.
- **Cause:** If `test.bat` spawns 6 Python processes, and each process calls a Rust kernel using a default 6-thread Rayon pool, the system attempts to manage **36 threads on 6 physical cores**.
- **Result:** Context-switching overhead dominated execution time, increasing 4K processing from ~200ms to ~6000ms.
- **Remedy:** In a multi-process Python context, the Rust backend must either be forced to single-threaded mode (`RAYON_NUM_THREADS=1`) or use a global thread-pool manager.

## 2. FFI Boundary & Memory Granularity
**Finding:** Offloading "simple" kernels (like RCT) to Rust can result in a net performance loss due to FFI overhead.
- **Detail:** RCT (Green-subtract) is memory-bound. Passing large NumPy arrays back and forth between Python and Rust consumes more time in memory management/ownership handoff than it saves in raw computation.
- **Strategy:** Rust should only be used for "thick" logic blocks (e.g., combining RCT + Prediction + Entropy Coding in one native call) to minimize the frequency of crossing the Python/Rust boundary.

## 3. Memory Safety & Edge Case Handling
**Finding:** C-style pointer arithmetic in Rust (`add(off)`) is extremely dangerous when input dimensions vary between RGB and RGBA.
- **Bug:** The `extract_channels_native` kernel attempted to write a default Alpha value (255) even when the input was 3-channel RGB. Since the Alpha output buffer was allocated as 0x0 for RGB, this caused an **Access Violation (Segfault)**.
- **Lesson:** Always use strict dimension gating (`if c == 4`) before any pointer-based writes, and prefer safe `ndarray` indexing over raw pointer math during the prototyping phase.

## 4. Numba Cache & Environment Conflicts (Windows/SHA1)
**Finding:** The presence of a PyO3-based native module in the same scope as Numba `@njit(cache=True)` functions can trigger environment-level crashes during process exit.
- **Symptom:** `_hashlib.UnsupportedDigestmodError: [digital envelope routines] unsupported`.
- **Cause:** Numba uses `hashlib.sha1` to generate cache keys. On Windows (OpenSSL 3.0), SHA1 is restricted. The import of `spx_rans` appears to alter the OpenSSL state or Python's hashing context, preventing Numba from completing its cleanup/caching phase.
- **Workaround:** Set `cache=False` for all Numba functions in modules that interact with the native backend on Windows.

## 5. Architectural Parity Requirement
**Finding:** Bit-perfect reconstruction (`MSE=0.0`) was achieved but requires explicit modular arithmetic in Rust (`wrapping_sub`) to match Numba/NumPy `uint8` overflow behavior.
- **Lesson:** Never assume standard arithmetic in Rust matches Python. Always use explicit `wrapping_` methods for bit-perfect codec parity.

## 6. Environmental Side-effects (Global Cache Invalidation)
**Finding:** The mere presence of a conflicting native module in `site-packages` can invalidate Numba caches project-wide, even if the module is not explicitly imported.
- **Symptom:** Benchmark `Warmup Time` increased from 4s to 25s, and `Avg Enc Time` increased 30x, even after reverting Python code changes.
- **Cause:** Numba's integrity check (SHA1 hashing) during the caching phase may scan or interact with the environment state. A DLL conflict (OpenSSL) introduced by the native module's installation can cause this check to fail silently, forcing re-compilation in every process/worker.
- **Action:** If native integration is paused or reverted, the module must be fully uninstalled (`pip uninstall`) to restore baseline Numba performance.

---
*Created on: 2026-04-22*
*Status: Native transition paused for architectural re-evaluation.*
