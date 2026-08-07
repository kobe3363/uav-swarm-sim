"""The 'square' obstacle kind: axis-aligned, fixed-size, and ZERO RNG draws --
plus a byte-identity guard that the pre-existing rectangle path is untouched.

Byte-identity for every SHIPPED config is carried primarily by the full
regression net (every obstacle-generating test uses non-'square' shapes and stays
green). These focused tests pin the mechanism: the new branch consumes 0 RNG
draws and is reachable ONLY via ``kind == "square"``, so the circle / rectangle /
polygon paths (and thus their RNG sequences) are byte-identical.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from uav_swarm_sim.planning.obstacle_generator import _unit_shape


def test_square_is_axis_aligned_exact_size_and_zero_draws():
    """A 'square' of side S is the axis-aligned S x S box about the origin and
    makes NO random draws (vs the rectangle branch's aspect + rotation)."""
    rng = np.random.default_rng(20260807)
    ref = np.random.default_rng(20260807)          # identical seed, never advanced
    poly = _unit_shape("square", 52.8, rng)

    # zero RNG draws inside _unit_shape -> rng is still pristine vs the reference
    assert rng.uniform() == ref.uniform()

    assert isinstance(poly, Polygon)
    minx, miny, maxx, maxy = poly.bounds
    assert maxx - minx == pytest.approx(52.8)
    assert maxy - miny == pytest.approx(52.8)       # unit aspect (a true square)
    corners = list(poly.exterior.coords)[:-1]       # drop the closing repeat
    assert len(corners) == 4                        # not a rotated / multi-vertex polygon
    for x, y in corners:
        assert abs(abs(x) - 26.4) < 1e-9            # +/- side/2, axis-aligned
        assert abs(abs(y) - 26.4) < 1e-9


def test_rectangle_path_still_consumes_exactly_two_draws():
    """Byte-identity guard: the rectangle branch must still consume exactly two
    uniform draws (aspect, then rotation), so inserting the 'square' branch did
    not perturb the RNG sequence of any existing (non-square) config."""
    rng = np.random.default_rng(12345)
    _unit_shape("rectangle", 50.0, rng)

    ref = np.random.default_rng(12345)
    ref.uniform(1.0, 3.0)       # aspect
    ref.uniform(0.0, 180.0)     # rotation
    assert rng.uniform() == ref.uniform()   # both advanced by exactly two draws
