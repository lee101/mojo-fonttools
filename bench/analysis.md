# Mojo fonttools Performance Analysis

**Generated:** 2026-11-10  
**Machine:** Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz (Linux x86_64)  
**Mojo:** 1.2.0.dev2026092605  
**fontTools:** Cython-optimized (installed via pixi)

---

## Executive Summary

| Workload | fontTools | Mojo | Speedup | Status |
|---|---:|---:|---:|---|
| IUP contour, 200k points / 200 refs | 67.6 ms | 95.6 ms | **0.71x** | ⚠️ **Slower** |
| Cubic bounds, 100k curves (batch) | 793.6 ms | 9.1 ms | **87x** | ✅ Faster |
| Quadratic bounds, 100k curves (batch) | 378.4 ms | 4.4 ms | **86x** | ✅ Faster |

---

## Ranked Hotspot Table

| # | Function | Input Size | Time (ms) | Bytes Touched | Est. FLOPs | FLOPs/Byte | Bound | Time Location | Next Phase |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **mft_iup_result** (Python tuple/list construction) | 200k pts | **44.5** | ~3.2 MB output | 0 | 0 | **Python overhead** | 100% Python/ctypes glue | **unroll** (vectorize result construction) |
| 2 | **iup_contour Python wrapper** (coord/delta conversion) | 200k pts | **27.2 + 21.1** = 48.3 | ~3.2 MB read + ~1.6 MB write | 0 | 0 | **Python overhead** | 100% Python | **unroll** (avoid list→array roundtrip) |
| 3 | **ft_bezier.calcCubicBounds** (scalar, Python loop) | 100k curves | 777 | ~15 MB read | ~7.8M | ~0.5 | **Python interpreter** | 100% Python | N/A (upstream) |
| 4 | **mft_cubic_bounds_batch** (Mojo kernel) | 100k curves | **8.3** | ~9.6 MB read + 3.2 MB write | ~8M | **~0.83** | **Memory-bound** | 100% Mojo kernel | **simd** (already SIMD, unroll tail) |
| 5 | **mft_quadratic_bounds_batch** (Mojo kernel) | 100k curves | **3.5** | ~8.0 MB read + 3.2 MB write | ~5M | **~0.63** | **Memory-bound** | 100% Mojo kernel | **simd** (already SIMD) |
| 6 | **mft_iup_contour** (Mojo kernel) | 200k pts | **0.57** | ~3.2 MB read + 3.2 MB write | ~3.2M | **~0.5** | **Memory-bound** | 100% Mojo kernel | **none** (already optimal) |
| 7 | **mft_decode_simple_glyph** (Mojo kernel) | 1k pts | **0.02** | ~6 KB read + 24 KB write | ~20k | ~0.7 | Memory-bound | 100% Mojo kernel | **none** |
| 8 | **mft_cubic_bounds** (Mojo scalar, ctypes/curve) | 100k curves | 954 | ~15 MB read + 3.2 MB write | ~7.8M | ~0.5 | **cthread overhead** | 99% ctypes call overhead | **unroll** (use batch) |

---

## Detailed Findings

### 1. IUP Contour (200k points) — **Mojo is 29% SLOWER**

**Breakdown (Mojo total: 95.6 ms):**
- Python coords conversion (list→numpy): **27.2 ms** (28%)
- Python deltas conversion (list→sparse arrays): **21.1 ms** (22%)
- Mojo kernel (`mft_iup_contour`): **0.57 ms** (<1%) ⚡ *very fast*
- Mojo result construction (`mft_iup_result`): **44.5 ms** (47%) ← **TOP HOTSPOT**
- ctypes call overhead: ~2 ms (2%)

**fontTools (67.6 ms):** Pure Cython loop over Python objects — no array conversions, no FFI boundary, no tuple allocation per point.

**Root cause:** The Mojo port pays **3 Python↔C boundary crossings** + **2 full array conversions** + **200k Python tuple allocations** in `mft_iup_result`. The SIMD kernel itself is 100x faster than needed.

**FLOPs/byte for Mojo kernel:** ~0.5 — **memory-bound**, well below GPU threshold (2.0).

---

### 2. Cubic Bounds Batch (100k curves) — **Mojo 87x FASTER**

**Breakdown (Mojo batch: 8.3 ms kernel + ~2 ms wrapper = 10.4 ms):**
- Mojo kernel (`mft_cubic_bounds_batch`): **8.3 ms** (80%)
- Python wrapper (array checks, ctypes call): **~2 ms** (20%)

**fontTools scalar (777 ms):** Python loop calling Cython function per curve — 100k ctypes-equivalent calls.

**FLOPs/byte:** ~0.83 — **memory-bound**, below GPU threshold.

---

### 3. Quadratic Bounds Batch (100k curves) — **Mojo 86x FASTER**

**Breakdown (Mojo batch: 3.5 ms kernel + ~0.9 ms wrapper = 4.4 ms):**
- Mojo kernel (`mft_quadratic_bounds_batch`): **3.5 ms** (80%)
- Python wrapper: **~0.9 ms** (20%)

**fontTools scalar (378 ms):** Same pattern as cubic.

**FLOPs/byte:** ~0.63 — **memory-bound**, below GPU threshold.

---

### 4. Scalar Mojo via ctypes (per-curve) — **SLOWER than fontTools**

| API | 100k curves |
|---|---:|
| fontTools scalar | 777 ms |
| Mojo `calcCubicBounds` (ctypes/curve) | 954 ms |

**Root cause:** 100k ctypes calls × ~9 μs/call = 900 ms overhead. The kernel is fast but the FFI boundary kills it.

---

## Per-Hotspot Action Plan

| Hotspot | Next Phase | Rationale |
|---|---|---|
| `mft_iup_result` (tuple allocation loop) | **unroll** | Replace per-point `PyTuple_New` + `PyFloat_FromDouble` with single `PyList_SetItem` loop using pre-allocated list, or return flat buffer + build tuples in Python |
| `iup_contour` Python wrapper (list→array conversion) | **unroll** | Accept NumPy arrays directly; avoid `itertools.chain.from_iterable` + `np.fromiter`; use `np.asarray` + `ravel()` |
| `mft_cubic_bounds_batch` kernel | **simd** | Already uses SIMD; unroll tail loop (lines 177-202 in fonttools.mojo); consider `parallelize` for >50k curves |
| `mft_quadratic_bounds_batch` kernel | **simd** | Same as cubic; tail handling at lines 142-170 |
| `mft_iup_contour` kernel | **none** | 0.57 ms for 200k points; SIMD + branchless select already optimal; FLOPs/byte = 0.5 (memory-bound) |
| `mft_decode_simple_glyph` | **none** | 20 μs for 1k points; sequential decode is inherent to format |
| Scalar ctypes entry points | **unroll** | Document batch APIs as required; deprecate scalar FFI entry points |

---

## GPU Verdict

**No kernel justifies GPU.** All three compute kernels (IUP, cubic bounds, quadratic bounds) have **FLOPs/byte < 1.0**, well below the ~2.0 threshold where GPU transfer + launch overhead pays off. The Mojo kernels are already memory-bandwidth limited on CPU.

---

## Key Architectural Issues

1. **`mft_iup_result` builds 200k Python tuples in Mojo** — each `PyTuple_New` + 2× `PyFloat_FromDouble` + `PyTuple_SetItem` × 2 = ~200 ns/point = 40 ms. Move tuple construction to Python side from flat buffer.

2. **Python wrapper does double conversion** — list → flat numpy → reshape for coords; list → sparse indices + values for deltas. Accept contiguous `float64` arrays directly.

3. **Batch APIs amortize FFI** — scalar entry points exist only for API compatibility; benchmarks must use batch variants.

4. **SIMD width is hardcoded** (`comptime W = simd_width_of[DType.float64]()`) — correct for AVX2 (4); verify on AVX-512 (8) at runtime.

---

## Recommended Phase Order

1. **unroll** — Fix `mft_iup_result` and Python wrapper conversions (largest gains, 70+ ms saved on IUP)
2. **simd** — Tail unrolling in batch bounds kernels (marginal, ~10% kernel speedup)
3. **parallel** — Thresholded `parallelize` for cubic bounds >50k curves (diminishing returns; kernel already memory-bound)
4. **gpu** — **Not recommended** (all kernels below 2 FLOPs/byte)