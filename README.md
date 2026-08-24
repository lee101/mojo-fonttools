# mojo-fonttools

`mojo-fonttools` is a standalone, open-source TrueType toolkit with compute kernels written
in Mojo and a Python API shaped after [fontTools](https://github.com/fonttools/fonttools).
It targets the parts of font processing where native code is useful: decoding packed
TrueType outlines, calculating Bézier geometry, interpolating variable-font deltas, and
closing and rewriting composite glyphs during subsetting.

fontTools is a development dependency used for parity tests and benchmarks. It is not
imported by the runtime package.

## Covered subset

- Read-only TrueType SFNT parsing through `mojo_fonttools.ttLib.TTFont`: table directory,
  `head`, `maxp`, `hhea`, `hmtx`, long and short `loca`, `glyf`, `post` format 2, and
  `cmap` formats 4 and 12.
- The commonly used `TTFont` methods `keys`, `getTableData`, `getGlyphOrder`,
  `getGlyphID`, `getGlyphName`, `getReverseGlyphMap`, `getBestCmap`, `getGlyphSet`,
  `save`, and the context-manager protocol.
- Packed simple-glyph coordinate and flag decoding in Mojo. Simple quadratic contours and
  XY-positioned/scaled composite components implement the pen protocol.
- `mojo_fonttools.subset.Options`, `Subsetter.populate`, and `Subsetter.subset`, plus the
  lower-level `subset_glyf`. Composite closure, component glyph-ID rewriting, `loca`,
  `hmtx`, `maxp`, `hhea`, `cmap`, `post`, table checksums, and the SFNT checksum adjustment
  are rebuilt. Of the `Options` fields, only `notdef_glyph` changes behavior; the other
  familiar fields are compatibility placeholders.
- fontTools-compatible outline geometry functions: quadratic/cubic point evaluation,
  bounds, parameter conversion, splitting at `t`, and approximate and measured arc length.
- `calcQuadraticBoundsBatch` and `calcCubicBoundsBatch` for contiguous `(n, 3, 2)` and
  `(n, 4, 2)` arrays.
- fontTools-compatible `varLib.iup.iup_contour` and `iup_delta`.

This is deliberately not a complete fontTools replacement. CFF/CFF2, WOFF/WOFF2, TTC,
variation table parsing, color fonts, XML/TTX, OpenType layout closure, point-attached
composites, table compilation/editing outside the subsetter, and fontTools' pens not listed
above are not covered. The subsetter preserves TrueType hinting programs but drops layout
and other glyph-indexed tables; it is intended for outline/cmap subsets, not shaping-aware
production subsetting.

## Install

The checked-in Pixi manifest pins the verified Mojo nightly and supplies Python, NumPy,
fontTools, and pytest:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-fonttools.so`. Run commands through Pixi so that
`python/` is on `PYTHONPATH`.

## Usage

```python
from mojo_fonttools import TTFont
from mojo_fonttools.subset import Subsetter

font = TTFont("input.ttf")
print(font["head"].unitsPerEm)
print(font.getBestCmap().get(ord("A")))

subsetter = Subsetter()
subsetter.populate(text="AÁ")
subsetter.subset(font)
font.save("subset.ttf")
```

The geometry functions keep fontTools' names and signatures:

```python
from mojo_fonttools.misc.bezierTools import calcCubicBoundsBatch
import numpy as np

curves = np.array(
    [[(0, 0), (25, 100), (75, 100), (100, 0)]],
    dtype=np.float64,
)
print(calcCubicBoundsBatch(curves))
# [[  0.   0. 100.  75.]]
```

## Benchmarks

Measured on 2026-08-24 with an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86-64. Values are
medians from `pixi run bench`; lower time is better and speedup is
`fontTools time / Mojo time`.

| Workload | fontTools | Mojo | Speedup |
|---|---:|---:|---:|
| IUP contour, 200k points / 200 refs | 67.572 ms | 82.290 ms | 0.82x |
| Cubic bounds, 100k curves (batch) | 755.781 ms | 9.048 ms | 83.53x |

The list-shaped IUP API flattens coordinates directly into one interleaved buffer and passes
only the sparse reference deltas. Contiguous NumPy coordinate buffers cross the FFI boundary
without a copy. Result tuples are constructed in one native CPython-API loop instead of
materializing two Python lists and zipping them. Python input and object construction still
dominate this workload; the SIMD interpolation kernel itself takes under one millisecond in
profiling. The batch bounds API amortizes the FFI cost across 100,000 curves and is
substantially faster. Scalar geometry calls also pay a ctypes call per curve, so use the
batch functions for throughput.

No parallel or GPU path is enabled. Thresholded CPU parallelization made the 200,000-point
IUP workload slower because its SIMD kernel is too short to repay thread-launch overhead.
The available batched kernels are also below the roughly two-flops-per-byte threshold once
input and output traffic is counted, so GPU transfer and launch overhead would not be
justified.

## How it works

Mojo compiles one source unit into a shared library. Python loads it with `ctypes`; arrays
remain owned by NumPy and cross the C ABI as integer addresses. Mojo reconstructs typed
mutable pointers using `AnyOrigin[mut=True]` and writes into caller-allocated contiguous
buffers, so the native layer performs no allocation and exposes no ownership protocol.

SFNT records remain big-endian byte strings in Python. The Mojo glyph decoder expands
TrueType's repeated flags and signed coordinate deltas directly into separate contiguous
`float64` X/Y arrays and a `uint8` semantic-flags array. The subsetter performs graph
closure over composite references, assigns a dense glyph order, patches component IDs,
then rebuilds and checksums a minimal valid TrueType SFNT.
