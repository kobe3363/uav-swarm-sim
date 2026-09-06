"""T1-T8: closed-form components, once-only accounting and the D-2 contract."""
from dataclasses import fields, replace
import math

import pytest
from shapely.geometry import GeometryCollection, LineString, Polygon, box

from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.infrastructure.enums import ManeuverType as M
from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.physical_model.vertical_segments import landing_profile
from uav_swarm_sim.planning.energy_balance import (
    EnergyBalanceStatus as S, estimate_fast, remaining_work_geometry,
)


def test_t1_every_fast_component(energy_case):
    c = energy_case
    e = estimate_fast(c.ctx, c.drone, c.zone, c.raster)
    chord = 220 * math.sqrt(400**2 + 40**2) / 12
    turns = 240 * (math.sqrt(24000) / 40 - 1) * 40 / 12
    demand = 15000 + turns + 900
    rth = chord + 240 * (2 * math.pi - 3 * math.atan2(40, 400)) + 10000
    budget = 350000 - (chord + rth + 18000)
    expected = dict(e_level_j=360000, e_takeoff_deducted_j=10000, e_remaining_j=350000,
                    e_ferry_j=chord, e_strips_j=15000, e_connectors_j=turns,
                    e_camera_j=900, e_coverage_j=demand, e_rth_j=rth,
                    e_reserve_j=18000, budget_j=budget, demand_j=demand,
                    demand_budget_ratio=demand / budget, remaining_area_m2=24000,
                    n_strips=math.sqrt(24000) / 40)
    for key, value in expected.items():
        assert getattr(e, key) == pytest.approx(value, rel=1e-9), key
    assert e.status is S.FEASIBLE
    assert e.method == "fast" and e.drone_id == 0
    assert e.anchor_pose.as_xy() == pytest.approx((400, 60))
    assert e.exit_pose.as_xy() == pytest.approx((400, 60))
    assert e.anchor_pose.heading == pytest.approx(math.atan2(40, 400), rel=1e-9)


def test_t2_takeoff_paid_once(energy_case):
    c = energy_case
    before = estimate_fast(c.ctx, c.drone, c.zone, None)
    after = estimate_fast(c.ctx, replace(c.drone, airborne=True, level_j=350000), c.zone, None)
    assert after.e_takeoff_deducted_j == pytest.approx(0)
    assert before.e_takeoff_deducted_j == pytest.approx(400 * 100 / 4)
    assert before.budget_j == pytest.approx(after.budget_j, rel=1e-9)


def test_t3_reserve_paid_once(energy_case):
    c = energy_case
    e = estimate_fast(c.ctx, c.drone, c.zone, None)
    assert e.e_reserve_j == pytest.approx(0.05 * 360000, rel=1e-9)
    assert e.budget_j + e.e_ferry_j + e.e_rth_j + e.e_reserve_j == pytest.approx(350000, rel=1e-9)


def test_t4_landing_paid_once(energy_case):
    c = energy_case
    e = estimate_fast(c.ctx, c.drone, c.zone, None)
    return_path = c.motion.plan(e.anchor_pose, c.base, M.CRUISE)
    assert e.e_rth_j - c.em.path_energy(return_path) == pytest.approx(
        landing_profile(c.spec, c.em, 100).energy_j, rel=1e-9)


@pytest.mark.parametrize("deficit", [0, 2000])
def test_t5_nonpositive_budget_is_explicit(energy_case, deficit):
    c = energy_case
    chord = 220 * math.hypot(400, 40) / 12
    rth = chord + 240 * (2 * math.pi - 3 * math.atan2(40, 400)) + 10000
    expenses = chord + rth + 18000
    drone = replace(c.drone, airborne=True, level_j=expenses - deficit)
    e = estimate_fast(c.ctx, drone, c.zone, None)
    assert e.budget_j == pytest.approx(-deficit, abs=1e-9)
    assert e.status is S.BUDGET_NONPOSITIVE
    assert e.demand_budget_ratio is None
    assert all(math.isfinite(getattr(e, f.name)) for f in fields(e)
               if isinstance(getattr(e, f.name), float))


def test_t6a_centroid_component_symmetry_and_asymmetric_budget(energy_case):
    c = energy_case
    a = estimate_fast(c.ctx, c.drone, c.zone, None)
    mirror_rth = RthCalculator(c.em, c.motion, c.spec, c.cfg.rth,
                               replace(c.base, heading=math.pi), 100.0)
    mirror_ctx = replace(c.ctx, return_energy=lambda p, alt: mirror_rth.return_energy(p, altitude_m=alt))
    mirror_drone = replace(c.drone, pose=replace(c.drone.pose, heading=math.pi))
    mirror_zone = replace(c.zone, polygon=box(-500, 0, -300, 120),
                          entry_pose=replace(c.zone.entry_pose, heading=math.pi))
    mirror = estimate_fast(mirror_ctx, mirror_drone, mirror_zone, None)
    distant = estimate_fast(c.ctx, c.drone, replace(c.zone, polygon=box(700, 0, 900, 120)), None)
    for field in fields(a):
        if isinstance(getattr(a, field.name), float):
            assert getattr(mirror, field.name) == pytest.approx(getattr(a, field.name), rel=1e-9)
    assert mirror.status is a.status
    assert distant.demand_budget_ratio > a.demand_budget_ratio


def test_t7_monotone_area_and_distance_at_fixed_anchor(energy_case):
    c = energy_case
    areas = [box(400 - w / 2, 60 - w / 2, 400 + w / 2, 60 + w / 2) for w in (40, 80, 160)]
    estimates = [estimate_fast(c.ctx, c.drone, replace(c.zone, polygon=g), None) for g in areas]
    assert all(e.anchor_pose.as_xy() == pytest.approx((400, 60)) for e in estimates)
    assert all(a.demand_j <= b.demand_j for a, b in zip(estimates, estimates[1:]))
    estimates = [estimate_fast(c.ctx, replace(c.drone, pose=Pose(x, 60, 0)), c.zone, None)
                 for x in (300, 200, 100)]
    assert all(a.e_ferry_j <= b.e_ferry_j and a.demand_budget_ratio <= b.demand_budget_ratio
               for a, b in zip(estimates, estimates[1:]))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_t8_nonfinite_level_rejected(energy_case, value):
    c = energy_case
    with pytest.raises(ValueError, match="level_j must be finite"):
        estimate_fast(c.ctx, replace(c.drone, level_j=value), c.zone, None)


def test_nonfinite_component_rejected(energy_case):
    c = energy_case
    with pytest.raises(ValueError, match="e_rth_j must be finite"):
        estimate_fast(replace(c.ctx, return_energy=lambda p, a: math.inf), c.drone, c.zone, None)


def test_empty_remaining_work_and_polygon_filter(energy_case):
    c = energy_case
    geometry = GeometryCollection([LineString([(0, 0), (1, 1)]), box(0, 0, 1, 1)])
    assert remaining_work_geometry(geometry, None).area == pytest.approx(1)
    empty = estimate_fast(c.ctx, c.drone, replace(c.zone, polygon=Polygon()), None)
    assert empty.demand_j == pytest.approx(0)
    assert empty.demand_budget_ratio == pytest.approx(0)
    assert empty.status is S.FEASIBLE
    assert empty.anchor_pose == c.zone.entry_pose
