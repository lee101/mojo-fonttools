from __future__ import annotations

import os
import platform
import statistics
import sys
import time

import numpy as np
from fontTools.misc import bezierTools as ft_bezier
from fontTools.varLib.iup import iup_contour as ft_iup_contour

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

from mojo_fonttools.misc import bezierTools as mojo_bezier
from mojo_fonttools.varLib.iup import iup_contour as mojo_iup_contour


def measure(function, repeats=5):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def cpu_name():
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    rng = np.random.default_rng(2026)
    coords = np.cumsum(rng.integers(-20, 30, size=(200_000, 2)), axis=0).tolist()
    deltas = [None] * len(coords)
    for index in range(0, len(coords), 1000):
        deltas[index] = (index % 31 - 15, index % 23 - 11)

    cubic_array = rng.normal(size=(100_000, 4, 2))
    cubic_points = [tuple(point) for row in cubic_array for point in row]

    def ft_bounds():
        for i in range(0, len(cubic_points), 4):
            ft_bezier.calcCubicBounds(*cubic_points[i : i + 4])

    def mojo_bounds():
        mojo_bezier.calcCubicBoundsBatch(cubic_array)

    workloads = [
        (
            "IUP contour, 200k points / 200 refs",
            lambda: ft_iup_contour(deltas, coords),
            lambda: mojo_iup_contour(deltas, coords),
            5,
        ),
        (
            "Cubic bounds, 100k curves (batch)",
            ft_bounds,
            mojo_bounds,
            3,
        ),
    ]
    rows = []
    for name, upstream, mojo, repeats in workloads:
        upstream_time = measure(upstream, repeats)
        mojo_time = measure(mojo, repeats)
        rows.append((name, upstream_time * 1000, mojo_time * 1000, upstream_time / mojo_time))

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| Workload | fontTools | Mojo | Speedup |")
    print("|---|---:|---:|---:|")
    for name, upstream_ms, mojo_ms, speedup in rows:
        print(f"| {name} | {upstream_ms:.3f} ms | {mojo_ms:.3f} ms | {speedup:.2f}x |")


if __name__ == "__main__":
    main()
