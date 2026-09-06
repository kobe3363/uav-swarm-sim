"""Analytic checks for EXP-02 persistent raster coverage truth."""
from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon, box

from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.planning.coverage_raster import CoverageRaster


def test_target_and_plannable_masks_use_raw_and_buffered_obstacles():
    area = box(0.0, 0.0, 100.0, 60.0)
    raw = box(40.0, 20.0, 60.0, 40.0)
    raster = CoverageRaster(area.difference(raw), area.difference(raw.buffer(10.0)), 10.0)

    assert raster.target_area_m2 == pytest.approx(5600.0)
    analytic_plannable_m2 = 6000.0 - (400.0 + 800.0 + 100.0 * math.pi)
    assert raster.plannable_area_m2 == pytest.approx(analytic_plannable_m2, abs=1.0)
    assert raster.target_cell_count == 56
    assert raster.plannable_cell_count == 48


def test_partial_segment_and_skipped_remainder_credit_only_flown_footprint():
    raster = CoverageRaster(box(0.0, 0.0, 100.0, 20.0), box(0.0, 0.0, 100.0, 20.0), 5.0)

    raster.record_segment(Pose(0.0, 10.0, 0.0), Pose(50.0, 10.0, 0.0), 10.0, 10.0)

    assert raster.target_coverage_frac == pytest.approx(0.275)
    assert raster.plannable_coverage_frac == pytest.approx(0.275)


def test_overlap_is_idempotent_and_plan_replacement_cannot_clear_coverage():
    raster = CoverageRaster(box(0.0, 0.0, 100.0, 20.0), box(0.0, 0.0, 100.0, 20.0), 5.0)
    segment = (Pose(0.0, 10.0, 0.0), Pose(50.0, 10.0, 0.0), 10.0, 10.0)
    raster.record_segment(*segment)
    before = raster.target_coverage_frac

    raster.record_segment(*segment)  # second UAV or a re-flown replacement plan

    assert raster.target_coverage_frac == pytest.approx(before)


def test_safe_flight_credits_target_cells_inside_clearance_buffer():
    area = box(0.0, 0.0, 100.0, 60.0)
    raw = box(40.0, 20.0, 60.0, 40.0)
    raster = CoverageRaster(area.difference(raw), area.difference(raw.buffer(10.0)), 2.0)

    # The UAV centre stays left of the clearance buffer (x < 30), while the
    # cross-track footprint reaches five metres into that visible buffer band.
    raster.record_segment(
        Pose(25.0, 25.0, math.pi / 2.0),
        Pose(25.0, 35.0, math.pi / 2.0),
        20.0,
        10.0,
    )

    assert raster.target_covered_area_m2 > raster.plannable_covered_area_m2


@pytest.mark.parametrize("cell_m", [2.5, 5.0])
def test_dt_partition_and_grid_size_preserve_exact_straight_sweep(cell_m):
    whole = CoverageRaster(
        box(0.0, 0.0, 100.0, 20.0), box(0.0, 0.0, 100.0, 20.0), cell_m
    )
    split = CoverageRaster(
        box(0.0, 0.0, 100.0, 20.0), box(0.0, 0.0, 100.0, 20.0), cell_m
    )
    whole.record_segment(Pose(0.0, 10.0, 0.0), Pose(100.0, 10.0, 0.0), 10.0, 10.0)
    for start in range(0, 100, 10):
        split.record_segment(
            Pose(float(start), 10.0, 0.0),
            Pose(float(start + 10), 10.0, 0.0),
            10.0,
            10.0,
        )

    assert whole.target_coverage_frac == pytest.approx(0.5)
    assert split.target_coverage_frac == pytest.approx(whole.target_coverage_frac)


def test_positive_area_empty_plan_starts_with_zero_coverage():
    raster = CoverageRaster(box(0.0, 0.0, 10.0, 10.0), box(0.0, 0.0, 10.0, 10.0), 2.0)

    assert raster.target_coverage_frac == pytest.approx(0.0)
    assert raster.plannable_coverage_frac == pytest.approx(0.0)
    assert raster.uncovered_plannable_geometry.area == pytest.approx(100.0)


def test_empty_target_is_a_complete_zero_cell_raster():
    empty = Polygon()
    raster = CoverageRaster(empty, empty, 2.0)

    assert raster.target_cell_count == 0
    assert raster.plannable_cell_count == 0
    assert raster.target_coverage_frac == pytest.approx(1.0)
    assert raster.plannable_coverage_frac == pytest.approx(1.0)
    assert raster.uncovered_plannable_geometry.is_empty


def test_stationary_yaw_adds_no_continuous_sweep_coverage():
    raster = CoverageRaster(box(0.0, 0.0, 10.0, 10.0), box(0.0, 0.0, 10.0, 10.0), 2.0)
    pose = Pose(5.0, 5.0, 0.0)

    raster.record_segment(pose, pose, 10.0, 10.0)

    assert raster.target_coverage_frac == pytest.approx(0.0)


def test_excessive_bounding_grid_is_rejected_before_allocation():
    area = box(0.0, 0.0, 10_000.0, 10_000.0)

    with pytest.raises(ValueError, match="increase coverage.raster_cell_m"):
        CoverageRaster(area, area, 0.1)


@pytest.mark.parametrize(
    ("footprint_width_m", "footprint_length_m"),
    [
        (math.nan, 10.0),
        (math.inf, 10.0),
        (-math.inf, 10.0),
        (10.0, math.nan),
        (10.0, math.inf),
        (10.0, -math.inf),
    ],
)
def test_non_finite_footprint_dimensions_are_rejected(
    footprint_width_m, footprint_length_m
):
    raster = CoverageRaster(box(0.0, 0.0, 10.0, 10.0), box(0.0, 0.0, 10.0, 10.0), 2.0)

    with pytest.raises(ValueError, match="finite and > 0"):
        raster.record_segment(
            Pose(0.0, 5.0, 0.0),
            Pose(10.0, 5.0, 0.0),
            footprint_width_m,
            footprint_length_m,
        )
