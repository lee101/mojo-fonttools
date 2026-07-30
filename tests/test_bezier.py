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
