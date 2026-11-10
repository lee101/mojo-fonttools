import numpy as np
import pytest
from fontTools.misc import bezierTools as reference

from mojo_fonttools.misc import bezierTools as mojo


@pytest.mark.parametrize("seed", range(5))
def test_quadratic_geometry_parity(seed):
    rng = np.random.default_rng(seed)
    for _ in range(100):
        points = [tuple(point) for point in rng.normal(size=(3, 2)) * 1000]
        t = rng.random()
        assert mojo.quadraticPointAtT(*points, t) == pytest.approx(
            reference.quadraticPointAtT(*points, t), abs=1e-11
        )
        assert mojo.calcQuadraticBounds(*points) == pytest.approx(
            reference.calcQuadraticBounds(*points), abs=1e-10
        )
        assert mojo.approximateQuadraticArcLength(*points) == pytest.approx(
            reference.approximateQuadraticArcLength(*points), rel=1e-14
        )
        assert mojo.calcQuadraticArcLength(*points) == pytest.approx(
            reference.calcQuadraticArcLength(*points), rel=1e-12
        )


@pytest.mark.parametrize("seed", range(5))
def test_cubic_geometry_parity(seed):
    rng = np.random.default_rng(seed)
    for _ in range(100):
        points = [tuple(point) for point in rng.normal(size=(4, 2)) * 1000]
        t = rng.random()
        assert mojo.cubicPointAtT(*points, t) == pytest.approx(
            reference.cubicPointAtT(*points, t), abs=1e-10
        )
        assert mojo.calcCubicBounds(*points) == pytest.approx(
            reference.calcCubicBounds(*points), abs=1e-9
        )
        assert mojo.approximateCubicArcLength(*points) == pytest.approx(
            reference.approximateCubicArcLength(*points), rel=1e-14
        )
        assert mojo.calcCubicArcLength(*points) == pytest.approx(
            reference.calcCubicArcLength(*points), rel=1e-12
        )


def test_split_at_t_parity():
    q = ((0, 0), (50, 100), (100, 0))
    c = ((0, 0), (25, 100), (75, 100), (100, 0))
    assert np.asarray(mojo.splitQuadraticAtT(*q, 0.2, 0.7)) == pytest.approx(
        np.asarray(reference.splitQuadraticAtT(*q, 0.2, 0.7))
    )
    assert np.asarray(mojo.splitCubicAtT(*c, 0.2, 0.7)) == pytest.approx(
        np.asarray(reference.splitCubicAtT(*c, 0.2, 0.7))
    )


def test_parameter_conversion_parity():
    q = ((4, -2), (10, 20), (30, 5))
    c = ((4, -2), (10, 20), (30, 5), (-8, 3))
    assert mojo.calcQuadraticParameters(*q) == reference.calcQuadraticParameters(*q)
    assert mojo.calcCubicParameters(*c) == reference.calcCubicParameters(*c)


def test_batch_bounds_match_upstream():
    rng = np.random.default_rng(99)
    quadratics = rng.normal(size=(1000, 3, 2))
    cubics = rng.normal(size=(1000, 4, 2))
    expected_q = np.asarray([reference.calcQuadraticBounds(*row) for row in quadratics])
    expected_c = np.asarray([reference.calcCubicBounds(*row) for row in cubics])
    assert mojo.calcQuadraticBoundsBatch(quadratics) == pytest.approx(expected_q, abs=1e-12)
    assert mojo.calcCubicBoundsBatch(cubics) == pytest.approx(expected_c, abs=1e-12)


def _hull(computed, upstream_rows):
    expected = np.asarray(upstream_rows, dtype=np.float64).reshape(computed.shape)
    assert computed == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("count", [1, 2, 3, 5, 7, 8, 9, 33, 100])
def test_batch_bounds_lengths_match_upstream(count):
    rng = np.random.default_rng(2000 + count)
    quads = rng.normal(size=(count, 3, 2))
    cubics = rng.normal(size=(count, 4, 2))
    _hull(
        mojo.calcQuadraticBoundsBatch(quads),
        [reference.calcQuadraticBounds(*row) for row in quads],
    )
    _hull(
        mojo.calcCubicBoundsBatch(cubics),
        [reference.calcCubicBounds(*row) for row in cubics],
    )


def test_batch_bounds_degenerate_and_axis_aligned():
    rng = np.random.default_rng(4000)
    quads = rng.normal(size=(100, 3, 2))
    cubics = rng.normal(size=(100, 4, 2))
    base = rng.normal(size=(100, 2))
    for i in range(0, 100, 4):
        mode = (i // 4) % 4
        if mode == 0:
            quads[i] = [[0, 0], [50, 50], [100, 100]]
            cubics[i] = [[0, 0], [33, 33], [66, 66], [100, 100]]
        elif mode == 1:
            quads[i] = [[0, 0], [10, 0], [100, 0]]
            cubics[i] = [[0, 0], [10, 0], [30, 0], [100, 0]]
        elif mode == 2:
            quads[i] = [[0, 0], [-50, -100], [-100, 0]]
            cubics[i] = [[0, 0], [-25, -100], [-75, -100], [-100, 0]]
        else:
            quads[i] = [[0, 0], [0, 50], [0, 100]]
            cubics[i] = [[0, 0], [0, 33], [0, 66], [0, 100]]
        quads[i] = quads[i] + base[i]
        cubics[i] = cubics[i] + base[i]
    _hull(
        mojo.calcQuadraticBoundsBatch(quads),
        [reference.calcQuadraticBounds(*row) for row in quads],
    )
    _hull(
        mojo.calcCubicBoundsBatch(cubics),
        [reference.calcCubicBounds(*row) for row in cubics],
    )


def test_cubic_bounds_quartic_derivative_no_linear_term():
    # Downward parabola ((0,0), (-25,-100), (-75,-100), (-100,0)) shifted up by a
    # tiny epsilon: the y-derivative coefficients are a ~= 6e-16, b = 600,
    # c = -300.  Upstream treats |a| < 1e-10 as a linear equation and finds the
    # interior minimum t = 1/2; solving it as a quadratic loses the root and the
    # port reported yMin = epsilon instead of about -75.
    curve = ((0.0, 0.001), (-25.0, -99.999), (-75.0, -99.999), (-100.0, 0.001))
    assert reference.calcCubicBounds(*curve) == pytest.approx(
        (-100.0, -74.999, 0.0, 0.001), abs=1e-9
    )
    assert mojo.calcCubicBounds(*curve) == pytest.approx(
        reference.calcCubicBounds(*curve), abs=1e-9
    )


def test_batch_bounds_reject_wrong_shapes():
    with pytest.raises(ValueError, match="shape"):
        mojo.calcCubicBoundsBatch(np.zeros((4, 3, 2)))
    with pytest.raises(ValueError, match="shape"):
        mojo.calcQuadraticBoundsBatch(np.zeros((4, 4, 2)))
