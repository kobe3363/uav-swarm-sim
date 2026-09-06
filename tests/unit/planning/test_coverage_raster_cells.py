"""EXP-07 read-only cell accessor: consistency with the EXP-02 raster's own API.

Every expectation here is derived from the raster's PRE-EXISTING public surface
(``plannable_cell_count``, ``plannable_area_m2``, ``uncovered_plannable_geometry``,
``plannable_covered_area_m2``) or from geometry computed independently -- never
from the accessor itself.
"""
from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import box

from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.planning.coverage_raster import CoverageRaster


@pytest.fixture
def clipped_raster():
    """The pinned fixture from test_coverage_raster.py: a buffered obstacle cuts
    the plannable space, so some cells are genuinely clipped."""
    area = box(0.0, 0.0, 100.0, 60.0)
    raw = box(40.0, 20.0, 60.0, 40.0)
    return CoverageRaster(area.difference(raw), area.difference(raw.buffer(10.0)), 10.0)


def test_cell_count_matches_the_rasters_own_counter(clipped_raster):
    cells = clipped_raster.uncovered_plannable_cells()
    assert len(cells) == clipped_raster.plannable_cell_count == 48
    assert len(cells.geometries) == len(cells.areas_m2) == len(cells.surface_points)
    assert cells.indices.tolist() == list(range(48))


def test_cell_areas_sum_to_the_plannable_area(clipped_raster):
    cells = clipped_raster.uncovered_plannable_cells()
    assert float(cells.areas_m2.sum()) == pytest.approx(
        clipped_raster.plannable_area_m2, rel=1e-12
    )


def test_cell_union_equals_the_uncovered_work_geometry(clipped_raster):
    cells = clipped_raster.uncovered_plannable_cells()
    union = shapely.union_all(cells.geometries)
    difference = union.symmetric_difference(clipped_raster.uncovered_plannable_geometry)
    assert difference.area == pytest.approx(0.0, abs=1e-9)


def test_covered_cells_leave_the_set_consistently_with_the_covered_area(clipped_raster):
    total = clipped_raster.plannable_area_m2
    clipped_raster.record_segment(Pose(0.0, 5.0, 0.0), Pose(100.0, 5.0, 0.0), 10.0, 10.0)

    cells = clipped_raster.uncovered_plannable_cells()
    assert len(cells) < clipped_raster.plannable_cell_count
    # remaining + credited == the whole plannable area, with no double counting
    assert float(cells.areas_m2.sum()) + clipped_raster.plannable_covered_area_m2 == (
        pytest.approx(total, rel=1e-12)
    )
    # and the survivors are exactly the cells the raster still calls uncovered
    assert shapely.union_all(cells.geometries).symmetric_difference(
        clipped_raster.uncovered_plannable_geometry
    ).area == pytest.approx(0.0, abs=1e-9)


def test_surface_points_are_not_area_centroids_on_clipped_cells(clipped_raster):
    """The accessor exposes ``point_on_surface`` under its own name.

    This is the one difference no other test in the suite can catch: the
    CVT/ENERGY identity test would use the same wrong point on both arms, and on
    an unclipped square grid the two points coincide exactly. A partitioner that
    used ``surface_points`` for the Lloyd centroid update would converge to
    something that is not a CVT, silently.
    """
    cells = clipped_raster.uncovered_plannable_cells()
    centroids = shapely.centroid(cells.geometries)
    offset = np.hypot(
        shapely.get_x(centroids) - shapely.get_x(cells.surface_points),
        shapely.get_y(centroids) - shapely.get_y(cells.surface_points),
    )
    # unclipped interior cells agree exactly; the boundary-clipped ones do not
    assert (offset <= 1e-9).sum() == 44
    assert (offset > 1e-9).sum() == 4
    assert offset.max() == pytest.approx(3.0345, abs=1e-3)


def test_accessor_is_a_pure_read(clipped_raster):
    before = clipped_raster.plannable_coverage_frac
    first = clipped_raster.uncovered_plannable_cells()
    second = clipped_raster.uncovered_plannable_cells()
    assert clipped_raster.plannable_coverage_frac == before
    assert len(first) == len(second)
    assert np.array_equal(first.areas_m2, second.areas_m2)


def test_empty_target_yields_an_empty_cell_set():
    from shapely.geometry import Polygon

    cells = CoverageRaster(Polygon(), Polygon(), 5.0).uncovered_plannable_cells()
    assert len(cells) == 0 and len(cells.geometries) == 0
