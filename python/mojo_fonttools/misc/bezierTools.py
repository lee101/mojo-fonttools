from __future__ import annotations

import math

import numpy as np

from .._lib import addr, lib


def _coords(*points):
    return [float(v) for point in points for v in point]


def quadraticPointAtT(pt1, pt2, pt3, t):
    result = np.empty(2, dtype=np.float64)
    lib().mft_quadratic_point(
        *_coords(pt1, pt2, pt3), float(t), addr(result, np.float64, writable=True)
    )
    return tuple(result)


def cubicPointAtT(pt1, pt2, pt3, pt4, t):
    result = np.empty(2, dtype=np.float64)
    lib().mft_cubic_point(
        *_coords(pt1, pt2, pt3, pt4), float(t), addr(result, np.float64, writable=True)
    )
    return tuple(result)


def linePointAtT(pt1, pt2, t):
    return (pt1[0] + (pt2[0] - pt1[0]) * t, pt1[1] + (pt2[1] - pt1[1]) * t)


def calcQuadraticBounds(pt1, pt2, pt3):
    result = np.empty(4, dtype=np.float64)
    lib().mft_quadratic_bounds(
        *_coords(pt1, pt2, pt3), addr(result, np.float64, writable=True)
    )
    return tuple(result)


def calcCubicBounds(pt1, pt2, pt3, pt4):
    result = np.empty(4, dtype=np.float64)
    lib().mft_cubic_bounds(
        *_coords(pt1, pt2, pt3, pt4), addr(result, np.float64, writable=True)
    )
    return tuple(result)


def calcQuadraticBoundsBatch(points):
    values = np.ascontiguousarray(points, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 2):
        raise ValueError("points must have shape (n, 3, 2)")
    result = np.empty((len(values), 4), dtype=np.float64)
    if len(values):
        lib().mft_quadratic_bounds_batch(
            addr(values, np.float64),
            len(values),
            addr(result, np.float64, writable=True),
        )
    return result


def calcCubicBoundsBatch(points):
    values = np.ascontiguousarray(points, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 2):
        raise ValueError("points must have shape (n, 4, 2)")
    result = np.empty((len(values), 4), dtype=np.float64)
    if len(values):
        lib().mft_cubic_bounds_batch(
            addr(values, np.float64),
            len(values),
            addr(result, np.float64, writable=True),
        )
    return result


def approximateQuadraticArcLength(pt1, pt2, pt3):
    return lib().mft_quadratic_arc_length(*_coords(pt1, pt2, pt3))


def approximateCubicArcLength(pt1, pt2, pt3, pt4):
    return lib().mft_cubic_arc_length(*_coords(pt1, pt2, pt3, pt4))


def calcQuadraticParameters(pt1, pt2, pt3):
    return (
        (pt1[0] - 2 * pt2[0] + pt3[0], pt1[1] - 2 * pt2[1] + pt3[1]),
        (2 * (pt2[0] - pt1[0]), 2 * (pt2[1] - pt1[1])),
        pt1,
    )


def calcCubicParameters(pt1, pt2, pt3, pt4):
    return (
        (
            -pt1[0] + 3 * pt2[0] - 3 * pt3[0] + pt4[0],
            -pt1[1] + 3 * pt2[1] - 3 * pt3[1] + pt4[1],
        ),
        (
            3 * pt1[0] - 6 * pt2[0] + 3 * pt3[0],
            3 * pt1[1] - 6 * pt2[1] + 3 * pt3[1],
        ),
        (3 * (pt2[0] - pt1[0]), 3 * (pt2[1] - pt1[1])),
        pt1,
    )


def calcQuadraticPoints(a, b, c):
    return c, (c[0] + b[0] / 2, c[1] + b[1] / 2), (
        a[0] + b[0] + c[0],
        a[1] + b[1] + c[1],
    )


def calcCubicPoints(a, b, c, d):
    return (
        d,
        (d[0] + c[0] / 3, d[1] + c[1] / 3),
        (d[0] + 2 * c[0] / 3 + b[0] / 3, d[1] + 2 * c[1] / 3 + b[1] / 3),
        (a[0] + b[0] + c[0] + d[0], a[1] + b[1] + c[1] + d[1]),
    )


def splitQuadraticAtT(pt1, pt2, pt3, *ts):
    a, b, c = calcQuadraticParameters(pt1, pt2, pt3)
    cuts = (0.0, *ts, 1.0)
    segments = []
    for t1, t2 in zip(cuts, cuts[1:]):
        delta = t2 - t1
        aa = (a[0] * delta**2, a[1] * delta**2)
        bb = ((2 * a[0] * t1 + b[0]) * delta, (2 * a[1] * t1 + b[1]) * delta)
        cc = (
            a[0] * t1**2 + b[0] * t1 + c[0],
            a[1] * t1**2 + b[1] * t1 + c[1],
        )
        segments.append(calcQuadraticPoints(aa, bb, cc))
    return segments


def splitCubicAtT(pt1, pt2, pt3, pt4, *ts):
    a, b, c, d = calcCubicParameters(pt1, pt2, pt3, pt4)
    cuts = (0.0, *ts, 1.0)
    segments = []
    for t1, t2 in zip(cuts, cuts[1:]):
        delta = t2 - t1
        aa = (a[0] * delta**3, a[1] * delta**3)
        bb = (
            (3 * a[0] * t1 + b[0]) * delta**2,
            (3 * a[1] * t1 + b[1]) * delta**2,
        )
        cc = (
            (2 * b[0] * t1 + c[0] + 3 * a[0] * t1**2) * delta,
            (2 * b[1] * t1 + c[1] + 3 * a[1] * t1**2) * delta,
        )
        dd = (
            a[0] * t1**3 + b[0] * t1**2 + c[0] * t1 + d[0],
            a[1] * t1**3 + b[1] * t1**2 + c[1] * t1 + d[1],
        )
        segments.append(calcCubicPoints(aa, bb, cc, dd))
    segments[0] = (pt1, *segments[0][1:])
    segments[-1] = (*segments[-1][:-1], pt4)
    return segments


def solveQuadratic(a, b, c, sqrt=math.sqrt):
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return []
        return [-c / b]
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    root = sqrt(disc)
    return [(-b - root) / (2 * a), (-b + root) / (2 * a)]


def calcQuadraticArcLength(pt1, pt2, pt3):
    p0, p1, p2 = complex(*pt1), complex(*pt2), complex(*pt3)
    d0, d1 = p1 - p0, p2 - p1
    d = d1 - d0
    n = d * 1j
    scale = abs(n)
    if scale == 0:
        return abs(p2 - p0)
    dot = lambda v1, v2: (v1 * v2.conjugate()).real
    original = dot(n, d0)
    if abs(original) < 1e-10:
        if dot(d0, d1) >= 0:
            return abs(p2 - p0)
        aa, bb = abs(d0), abs(d1)
        return (aa * aa + bb * bb) / (aa + bb)
    x0, x1 = dot(d, d0) / original, dot(d, d1) / original
    integral = lambda x: x * math.sqrt(x * x + 1) / 2 + math.asinh(x) / 2
    return abs(2 * (integral(x1) - integral(x0)) * original / (scale * (x1 - x0)))


def calcCubicArcLength(pt1, pt2, pt3, pt4, tolerance=0.005):
    def recurse(p0, p1, p2, p3):
        arch = abs(p0 - p3)
        box = abs(p0 - p1) + abs(p1 - p2) + abs(p2 - p3)
        if arch * (1 + 1.5 * tolerance) + 1e-9 >= box:
            return (arch + box) / 2
        mid = (p0 + 3 * (p1 + p2) + p3) * 0.125
        deriv = (p3 + p2 - p1 - p0) * 0.125
        return recurse(p0, (p0 + p1) / 2, mid - deriv, mid) + recurse(
            mid, mid + deriv, (p2 + p3) / 2, p3
        )

    return recurse(complex(*pt1), complex(*pt2), complex(*pt3), complex(*pt4))
