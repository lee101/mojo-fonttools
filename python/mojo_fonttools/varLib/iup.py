from __future__ import annotations

import itertools

import numpy as np

from .._lib import addr, lib, py_lib


def iup_contour(deltas, coords):
    if len(deltas) != len(coords):
        raise ValueError("deltas and coords must have equal length")
    n = len(coords)
    if not n:
        return []
    if isinstance(coords, np.ndarray):
        c = np.asarray(coords, dtype=np.float64)
    else:
        coord_values = itertools.chain.from_iterable(coords)
        flat_coords = np.fromiter(
            coord_values,
            dtype=np.float64,
            count=n * 2,
        )
        sentinel = object()
        if next(coord_values, sentinel) is not sentinel:
            raise ValueError("coordinates must be pairs")
        c = flat_coords.reshape(n, 2)
    if c.shape != (n, 2):
        raise ValueError("coordinates must be pairs")
    if not c.flags.c_contiguous:
        c = np.ascontiguousarray(c)

    if isinstance(deltas, np.ndarray) and deltas.dtype != object:
        dense_refs = np.asarray(deltas, dtype=np.float64)
        if dense_refs.shape != (n, 2):
            raise ValueError("deltas must be pairs")
        valid = ~np.isnan(dense_refs).any(axis=1)
        indices = np.flatnonzero(valid)
        refs = (
            np.ascontiguousarray(dense_refs).reshape(-1)
            if len(indices) == n
            else np.ascontiguousarray(dense_refs[valid]).reshape(-1)
        )
    else:
        ref_indices = []
        ref_deltas = []
        for index, value in enumerate(deltas):
            if value is not None:
                x, y = value
                if x == x and y == y:
                    ref_indices.append(index)
                    ref_deltas.extend((x, y))
        indices = np.asarray(ref_indices, dtype=np.int64)
        refs = np.asarray(ref_deltas, dtype=np.float64)

    result = np.empty((2, n), dtype=np.float64)
    lib().mft_iup_contour(
        addr(c, np.float64),
        n,
        addr(indices, np.int64) if len(indices) else 0,
        addr(refs, np.float64) if len(refs) else 0,
        len(indices),
        addr(result, np.float64, writable=True),
    )
    return py_lib().mft_iup_result(addr(result, np.float64), n)


def iup_delta(deltas, coords, ends):
    if len(deltas) != len(coords):
        raise ValueError("deltas and coords must have equal length")
    if list(ends) != sorted(ends):
        raise ValueError("contour endpoints must be sorted")
    expected = (ends[-1] + 1 if ends else 0) + 4
    if len(coords) != expected:
        raise ValueError("IUP outlines must include four phantom points")
    result = []
    start = 0
    for end in [*ends, len(coords) - 4, len(coords) - 3, len(coords) - 2, len(coords) - 1]:
        if end < start or end >= len(coords):
            raise ValueError("contour endpoints must be sorted and in range")
        result.extend(iup_contour(deltas[start : end + 1], coords[start : end + 1]))
        start = end + 1
    return result
