import numpy as np
import pytest
from fontTools.varLib.iup import iup_contour as reference_contour
from fontTools.varLib.iup import iup_delta as reference_delta

from mojo_fonttools.varLib.iup import iup_contour, iup_delta


@pytest.mark.parametrize(
    "deltas",
    [
        [None, None, None, None],
        [(2, -4), None, None, None],
        [(0, 0), None, (20, -10), None],
        [None, (5, 7), None, (-3, 2)],
    ],
)
def test_iup_contour_reference_vectors(deltas):
    coords = [(0, 0), (100, 20), (200, 80), (300, 40)]
    assert np.asarray(iup_contour(deltas, coords)) == pytest.approx(
        np.asarray(reference_contour(deltas, coords))
    )


def test_iup_random_parity():
    rng = np.random.default_rng(17)
    for n in (3, 10, 50, 200):
        coords = np.cumsum(rng.integers(-50, 100, size=(n, 2)), axis=0).tolist()
        deltas = [None] * n
        for index in rng.choice(n, max(1, n // 5), replace=False):
            deltas[index] = tuple(rng.integers(-30, 30, size=2))
        assert np.asarray(iup_contour(deltas, coords)) == pytest.approx(
            np.asarray(reference_contour(deltas, coords))
        )


@pytest.mark.parametrize(
    "deltas",
    [
        [None] * 7,
        [(3, -2)] + [None] * 6,
        [(3, -2), None, None, (-4, 9), None, (2, 1), None],
    ],
)
def test_iup_simd_tail_paths(deltas):
    coords = [(0, 0), (7, 12), (19, -3), (31, 18), (48, 4), (66, 27), (85, 9)]
    result = iup_contour(deltas, coords)
    assert all(isinstance(value, tuple) for value in result)
    assert np.asarray(result) == pytest.approx(
        np.asarray(reference_contour(deltas, coords))
    )


def test_iup_numpy_buffers():
    coords = np.array(
        [(0, 0), (12, 7), (25, 21), (39, 8), (54, 33), (70, 15), (87, 42)],
        dtype=np.float64,
    )
    deltas = np.full_like(coords, np.nan)
    deltas[[0, 3, 6]] = [(2, -1), (8, 5), (-4, 7)]
    reference_deltas = [
        None if np.isnan(value).any() else tuple(value) for value in deltas
    ]
    assert np.asarray(iup_contour(deltas, coords)) == pytest.approx(
        np.asarray(reference_contour(reference_deltas, coords))
    )
    assert np.asarray(iup_contour(deltas, coords[:, ::-1])) == pytest.approx(
        np.asarray(reference_contour(reference_deltas, coords[:, ::-1]))
    )


def test_iup_delta_multiple_contours_and_phantoms():
    coords = [
        (0, 0), (100, 0), (100, 100), (0, 100),
        (500, 0), (600, 100),
        (0, 0), (600, 0), (0, 900), (0, -250),
    ]
    deltas = [(0, 0), None, (20, 10), None, None, (5, -5), (7, 8), None, (0, 3), None]
    ends = [3, 5]
    assert np.asarray(iup_delta(deltas, coords, ends)) == pytest.approx(
        np.asarray(reference_delta(deltas, coords, ends))
    )
