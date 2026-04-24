# SPX v8.3.2 — Pre-Rust Code Review

**Scope:** All `.py` files in `core/`  
**Purpose:** Identify bugs, dead code, redundancy, wrong annotations, and design issues before Rust rewrite.  
**Date:** 2026-04-24

---

## Systematic Approach

Issues are grouped into five categories ranked by fix priority:

1. **BUG** — Functional correctness or silent misbehaviour
2. **DEAD CODE** — Functions, variables, parameters, or flags that can never be reached
3. **REDUNDANCY** — Duplicate logic, double allocation, or double initialisation
4. **WRONG ANNOTATION** — Incorrect comments, docstrings, or spec tables
5. **UNUSED IMPORT** — Symbols imported but never referenced

Each entry lists: file, location, description, and recommended action.

---

## 1. BUGS

### B-01 · `SPX_DISABLE_TEMPLATES` env var silently ignored in standard rANS path
**File:** `rans.py` — `build_pdf_tables_from_shards_core()` (line ~188)  
**Description:** `build_pdf_tables_from_shards_core` calls `_decide_shard_mode_core(..., disable_templates=False)` with the flag hardcoded to `False`. The env var check lives only in `rans_selector.decide_shard_mode()`, which is never called in the standard sharded encode path. As a result, `SPX_DISABLE_TEMPLATES=1` has no effect on Kodak/DIV2K/CLIC images — only on any direct caller of `decide_shard_mode`.  
**Fix:** Either read the env var inside `build_pdf_tables_from_shards` (non-JIT wrapper), or promote the check into `build_pdf_tables_from_shards_core` via an extra bool parameter threaded from the caller.

---

### B-02 · `fused_rct_p1_gray` allocates full-size zero arrays for unused rd/bd residual cache
**File:** `sharding.py` — `fused_rct_p1_gray()` (line ~392)  
**Description:**  
```python
empty_pad = np.zeros((h + 2, w + 2), dtype=np.uint8)
res_cached = (gr_res, empty_pad, empty_pad, a_res)
```
For a 1920×1080 image, each `empty_pad` is ~2 MB. Two are allocated (same object stored in two tuple slots, but one `np.zeros` call still runs). The bitplane grayscale encoder only accesses `res_cached[0]`; these arrays are never read.  
**Fix:** Replace with `np.empty((0, 0), dtype=np.uint8)` — consistent with how `rd_ch_p`/`bd_ch_p` are handled in `compress.py`.

---

## 2. DEAD CODE

### D-01 · `med_standard()` — never called
**File:** `predictor.py`  
**Description:** `selected_predictor` calls only `med_edge_tuned`. `med_standard` is an unreachable fallback left over from before the edge-tuned version was promoted. It is not exported in `__init__.py` and is not called from any test.  
**Fix:** Delete.

---

### D-02 · `BICC_ZIGZAG_LUT` — never imported or used
**File:** `predictor.py` (line ~119)  
**Description:** Defined as a module-level LUT but not imported by any other file. The inline comment says "Combined BICC Bias + ZigZag LUT for direct normalisation" but it was apparently superseded by the inline `ZIGZAG_LUT[uint8((int(v) - int(p)) & 0xFF)]` pattern used throughout `sharding.py`.  
**Fix:** Delete.

---

### D-03 · `ENABLE_DIAGNOSTICS` — never read
**File:** `common.py` (line ~84)  
**Description:** `ENABLE_DIAGNOSTICS: bool = False` is a module-level constant that is never imported or checked in any production or test file.  
**Fix:** Delete.

---

### D-04 · `force_mode` parameter in `compress_spx` — accepted but ignored
**File:** `compress.py` — `compress_spx()` signature (line ~118)  
**Description:** The parameter is received, but there is no `if force_mode is not None:` branch anywhere in the function. Its value is silently discarded.  
**Fix:** Delete the parameter, or implement it if the functionality is still intended.

---

### D-05 · `collect_freqs_jit()` — superseded, never called
**File:** `rans.py` (line ~157)  
**Description:** Single-shard histogram helper that was superseded by the parallel `collect_all_freqs_parallel`. Not imported anywhere outside the file, and not called within the file either.  
**Fix:** Delete.

---

### D-06 · `rans_decode_shards_parallel()` — defined but never imported or called
**File:** `rans.py` (line ~481)  
**Description:** A parallel shard decode dispatcher that mirrors the encode-side `rans_encode_shards_parallel`. The actual decode path in `codec.unpack_bitstream` uses a Python `ThreadPoolExecutor` calling `rans_decode_4way_core` per shard directly, and never imports `rans_decode_shards_parallel`.  
**Fix:** Delete, or replace the decode loop in `unpack_bitstream` with it (the latter is the better long-term choice).

---

### D-07 · FLAG_SIMPLE / FLAG_RAW / FLAG_PASSTHROUGH decode paths — compressor never sets these flags
**Files:** `decompress.py` (~lines 163, 179–191), `codec.py` (`unpack_bitstream` ~line 233)  
**Description:** Three legacy flags (`FLAG_SIMPLE=0x02`, `FLAG_RAW=0x04`, `FLAG_PASSTHROUGH=0x08`) are defined in `common.py` and checked in the decompressor, but `compress_spx` never sets any of them. The only flags the compressor sets are `FLAG_RGBA`, `FLAG_GRAYSCALE`, `FLAG_COLOR_GSUB`, `FLAG_BITPLANE`. The three decode branches are therefore unreachable from any SPX file produced by the current compressor.  
**Fix:** If these modes are permanently removed, delete the flag constants and their decode branches. If they are intended for future use, document that clearly.

---

### D-08 · `nsid` local variable in `decompress_spx` — assigned, never used
**File:** `decompress.py` (line ~173)  
**Description:** `nsid = profile.noise_shard_id` is assigned immediately after `profile = PROFILE_RGB` but is never referenced again in the function.  
**Fix:** Delete the assignment.

---

### D-09 · `extract_channels()` in `transform.py` — never called from main pipeline
**File:** `transform.py`  
**Description:** `extract_channels` performs G-Sub RCT + global histogram extraction. The production path performs this step inline inside `fused_rct_p1_rgb` in `sharding.py`. Neither `compress.py`, `decompress.py`, nor `rans_bitplane.py` import it. May still be used in `test_suite.py` — verify before deleting.  
**Fix:** Verify usage. If unused, delete.

---

### D-10 · `predict_2d_residuals()` in `transform.py` — likely dead
**File:** `transform.py` (line ~128)  
**Description:** The docstring says "Used primarily for Alpha and non-sharded grayscale modes," but Alpha residuals are now computed inline in `fused_rct_p1_rgb/gray`, and the grayscale non-sharded paths are themselves dead (see D-07). Not imported in any production file.  
**Fix:** Verify usage against `test_suite.py`. If unused in production, delete.

---

### D-11 · Grayscale standard-rANS path is unreachable
**File:** `compress.py` (line ~163), `sharding.py` (`shard_pass_2_gray_stateless`), `codec.py` (`pack_bitstream`)  
**Description:** `compress_spx` forces `use_bitplane=True` for all grayscale images unconditionally:
```python
use_bitplane = True if is_grayscale else _evaluate_coder_selection(...)
```
This means `execute_sharding_stateless` is never called for grayscale, and the grayscale branch inside `pack_bitstream` can never execute. The infrastructure for the grayscale sharded path (`shard_pass_2_gray_stateless`, grayscale branches in `unpack_bitstream`) exists but cannot be triggered.  
**Fix:** Either remove the grayscale sharded path entirely, or document the forced-bitplane decision explicitly and make the path reachable via a flag/env override.

---

### D-12 · `reconstruct_shards_rgb` `is_grayscale` parameter — always False when called
**File:** `sharding.py` — `reconstruct_shards_rgb()` (line ~443)  
**Description:** The function accepts `is_grayscale` and allocates `(1,1)` dummy arrays for `rd_rec`/`bd_rec` when True. But in `decompress.py`, grayscale takes the bitplane path (see D-11), so this parameter is always False in practice.  
**Fix:** Remove the `is_grayscale` parameter and the associated conditional allocations.

---

### D-13 · `calculate_channel_stats()` — verify usage
**File:** `sharding.py` (line ~79)  
**Description:** Imported in `compress.py` but does not appear to be called anywhere in the compress flow — `SpxResult` is constructed without invoking it. The function returns the mode of a histogram.  
**Fix:** Grep for callers. If only in `test_suite.py`, move it there or delete it from the public sharding API.

---

## 3. REDUNDANCY

### R-01 · `build_shard_map_universal_42()` called twice at module level
**File:** `sharding.py` (lines ~197–208)  
**Description:**  
```python
_s_lut_rgb, _i_lut_rgb, _d_lut_rgb = precompute_luts(..., build_shard_map_universal_42(), -1)
PROFILE_RGB = ShardProfile(..., shard_map=build_shard_map_universal_42(), ...)
```
The function is called twice, computing the same shard map. The result of the first call should be stored and reused.  
**Fix:**  
```python
_shard_map_rgb = build_shard_map_universal_42()
_s_lut_rgb, _i_lut_rgb, _d_lut_rgb = precompute_luts(..., _shard_map_rgb, -1)
PROFILE_RGB = ShardProfile(..., shard_map=_shard_map_rgb, ...)
```

---

### R-02 · `PROFILE_RGB` double-imported in `decompress.py`
**File:** `decompress.py` (lines ~37 and ~45)  
**Description:**  
```python
from .sharding import PROFILE_RGB, reconstruct_shards_rgb   # line 37
...
from .sharding import PROFILE_RGB                           # line 45 — duplicate
```
**Fix:** Merge into a single import statement.

---

### R-03 · Thread-local ZstdDecompressor duplicated between `decompress.py` and `codec.py`
**Files:** `decompress.py` (`zstandard_decompress` + `thread_local_decomp`), `codec.py` (`get_zstd_decomp`)  
**Description:** Two independent thread-local decompressor caches are maintained for the same purpose. `decompress.py` creates `thread_local_decomp.decomp` via `zstandard_decompress()`, while `codec.py` creates one via `get_zstd_decomp()`. The decompressor in `decompress.py` is used only for the `is_simple` path which is dead code (D-07). The `rans_bitplane.py` decompressors are also inline (`zstd.ZstdDecompressor()`).  
**Fix:** Once D-07 dead paths are removed, consolidate all decompressor access through `codec.get_zstd_decomp()`.

---

### R-04 · `env.verify_environment()` called twice at module import
**Files:** `compress.py` (line ~47), `decompress.py` (line ~49)  
**Description:** Both modules call `env.verify_environment()` at module level. When the package is imported normally via `__init__.py` (which imports both), the check runs twice.  
**Fix:** Add a module-level guard in `env.py`:
```python
_verified = False
def verify_environment() -> bool:
    global _verified
    if _verified: return True
    ...
    _verified = True
    return True
```

---

### R-05 · Local rANS constants in `_rans_decode_sharded_with_ref` duplicate module-level constants
**File:** `rans_bitplane.py` — `_rans_decode_sharded_with_ref()` (lines ~484–487)  
**Description:**  
```python
l_lower = np.uint64(1 << 31)   # duplicates ANS_L_LOWER
m_bits  = np.uint64(12)        # duplicates ANS_M_BITS
mask    = np.uint64((1 << 12) - 1)  # duplicates ANS_PRECISION - 1
```
`_rans_decode_sharded` (the sibling function) uses the module-level constants correctly. The inconsistency could cause a subtle mismatch if the constants are ever updated.  
**Fix:** Replace the three local definitions with the module-level `ANS_L_LOWER`, `ANS_M_BITS`, `ANS_PRECISION` constants.

---

### R-06 · Per-call magic table in `rans_encode_shards_parallel` — not unified with `_MAGIC_LUT`
**File:** `rans.py` — `rans_encode_shards_parallel()` (lines ~322–330)  
**Description:** The standard rANS encoder still allocates a `(num_shards, 256)` magic table per image:
```python
all_magics = np.empty((num_shards, 256), dtype=uint64)
for i in prange(num_shards):
    sf = all_sym_freqs[i]
    for s_idx in range(256):
        f_v = sf[s_idx]
        if f_v > 0: all_magics[i, s_idx] = uint64(0xFFFFFFFFFFFFFFFF) // f_v
```
The bitplane path was updated to use a 1D `_MAGIC_LUT` indexed by `f` value (32 KB, L1-resident). The standard path was not. For 42 shards × 256 symbols × 8 bytes = 86 KB + recomputation overhead.  
**Fix:** Replace `all_magics[i, s_idx]` lookups with `_MAGIC_LUT[f_v]` (import or replicate the 4097-entry LUT from `rans_bitplane.py`, or move it to `common.py`).

---

### R-07 · Inline `ZstdDecompressor()` per call in `rans_bitplane.py` decompressors
**File:** `rans_bitplane.py` — `decompress_bitplane_gray_sharded()` (line ~612), `_unpack_channel` inside `decompress_bitplane_rgb_sharded`  
**Description:** Both create a new `zstd.ZstdDecompressor()` object on every decode call instead of reusing a cached instance.  
**Fix:** Use `codec.get_zstd_decomp()` (thread-local cache) or replicate the thread-local pattern here.

---

## 4. WRONG ANNOTATIONS

### A-01 · Codec bitstream spec table: wrong flag hex values
**File:** `codec.py` (line ~23)  
**Description:** The bitstream spec comment states:
```
| 20     | uint32 | Flags     | RGBA(0x1), Gray(0x8), GSUB(0x10), BP(0x40)   |
```
Actual values from `common.py`:
- `FLAG_GRAYSCALE = 0x10` (not `0x8`)
- `FLAG_COLOR_GSUB = 0x20` (not `0x10`)

**Fix:**  
```
| 20     | uint32 | Flags     | RGBA(0x01), Simple(0x02), Raw(0x04), Pass(0x08), Gray(0x10), GSub(0x20), BP(0x40) |
```

---

### A-02 · "Mode 4-9" should be "Mode 4-33" in rans.py comments
**File:** `rans.py` — `compact_pdf_tables` docstring and inline comment in `build_pdf_tables_from_shards_core` (line ~198)  
**Description:** Multiple places reference "Modes 4-9" or "4-9 (Templates)" but there are 30 templates spanning Modes 4–33.  
**Fix:** Replace all occurrences of `4-9` with `4-33` in these comments.

---

### A-03 · `compact_pdf_tables` inline comment "Mode 3 (Empty) or 4-9 (Templates)"
**File:** `rans.py` (line ~231)  
```python
if mode >= 3: # Mode 3 (Empty) or 4-9 (Templates)
```
**Fix:** `# Mode 3 (Empty) or 4-33 (Templates)`

---

### A-04 · `_rans_decode_sharded_with_ref` local constant names obscure relationship to protocol
**File:** `rans_bitplane.py`  
**Description:** `l_lower`, `m_bits`, `mask` are clearly the same values as `ANS_L_LOWER`, `ANS_M_BITS`, `ANS_PRECISION-1` but have different names, making code review and Rust porting harder. Covered under R-05 but also an annotation issue.

---

### A-05 · `reconstruct_2d_channels` in `transform.py` — `Optional` parameter annotation for Numba JIT
**File:** `transform.py` (line ~152)  
**Description:** The signature uses `Optional[npt.NDArray[np.uint8]] = None` for `out`. Numba's `@njit` does not support Python's `Optional` union type in signatures at runtime; it works here only because Numba infers the type from the call site. This could break under stricter Numba compilation modes. The annotation is misleading.  
**Fix:** Use two separate JIT functions (with and without `out`), or remove the Numba decorator and keep it as a Python function (it's not on the hot path).

---

## 5. UNUSED IMPORTS

### I-01 · `List, Optional` unused in `rans.py`
**File:** `rans.py` (line ~41)  
```python
from typing import Tuple, List, Optional
```
Neither `List` nor `Optional` appears in any function signature or annotation in this file.  
**Fix:** Remove `List, Optional` from the import.

---

### I-02 · `traceback` imported inside `except` block in `decompress.py`
**File:** `decompress.py` (line ~238)  
```python
except Exception as e:
    logger.error(...)
    import traceback
    logger.debug(traceback.format_exc())
```
Runtime import inside an exception handler is an anti-pattern — it will fail silently if the module is unavailable and hides the dependency.  
**Fix:** Move `import traceback` to the top of the file.

---

## Summary Table

| ID   | Category         | File(s)                          | Severity |
|------|------------------|----------------------------------|----------|
| B-01 | Bug              | `rans.py`                        | High     |
| B-02 | Bug              | `sharding.py`                    | Medium   |
| D-01 | Dead Code        | `predictor.py`                   | Low      |
| D-02 | Dead Code        | `predictor.py`                   | Low      |
| D-03 | Dead Code        | `common.py`                      | Low      |
| D-04 | Dead Code        | `compress.py`                    | Medium   |
| D-05 | Dead Code        | `rans.py`                        | Low      |
| D-06 | Dead Code        | `rans.py`                        | Low      |
| D-07 | Dead Code        | `decompress.py`, `codec.py`      | Medium   |
| D-08 | Dead Code        | `decompress.py`                  | Low      |
| D-09 | Dead Code        | `transform.py`                   | Low      |
| D-10 | Dead Code        | `transform.py`                   | Low      |
| D-11 | Dead Code        | `compress.py`, `sharding.py`     | Medium   |
| D-12 | Dead Code        | `sharding.py`                    | Low      |
| D-13 | Dead Code        | `sharding.py`                    | Low      |
| R-01 | Redundancy       | `sharding.py`                    | Low      |
| R-02 | Redundancy       | `decompress.py`                  | Low      |
| R-03 | Redundancy       | `decompress.py`, `codec.py`      | Low      |
| R-04 | Redundancy       | `compress.py`, `decompress.py`   | Low      |
| R-05 | Redundancy       | `rans_bitplane.py`               | Medium   |
| R-06 | Redundancy       | `rans.py`                        | Medium   |
| R-07 | Redundancy       | `rans_bitplane.py`               | Low      |
| A-01 | Wrong Annotation | `codec.py`                       | Medium   |
| A-02 | Wrong Annotation | `rans.py`                        | Low      |
| A-03 | Wrong Annotation | `rans.py`                        | Low      |
| A-04 | Wrong Annotation | `rans_bitplane.py`               | Low      |
| A-05 | Wrong Annotation | `transform.py`                   | Low      |
| I-01 | Unused Import    | `rans.py`                        | Low      |
| I-02 | Unused Import    | `decompress.py`                  | Low      |

**Total: 28 issues** — 2 Bugs, 13 Dead Code, 7 Redundancy, 5 Wrong Annotation, 2 Unused Import

---

## Suggested Fix Order

1. **B-01** — Fix first; it's a silent functional divergence from documented behaviour.
2. **D-07 → D-11** — Removing the three legacy flags and the dead grayscale-sharded path clears the most clutter and simplifies the Rust port surface.
3. **R-05, R-06** — Constant/LUT unification; makes the Rust port consistent between the two rANS paths.
4. **D-01–D-06, D-08–D-10, D-12–D-13** — Mechanical deletions; safe, no logic change.
5. **R-01–R-04, R-07** — Minor housekeeping.
6. **A-01–A-05, I-01–I-02** — Annotation and import cleanup.
7. **B-02** — Low-risk memory fix; change `np.zeros((h+2, w+2))` to `np.empty((0,0))` for the two unused channels in grayscale pass1.
