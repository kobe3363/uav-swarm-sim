"""T9-T15: real HolonomicModel routes, persistent work and explicit feasibility."""
from dataclasses import fields, replace
import math

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.planning.obstacle_generator import Obstacle
from uav_swarm_sim.infrastructure.enums import ManeuverType as M
from uav_swarm_sim.planning.coverage_path import boustrophedon
from uav_swarm_sim.planning.energy_balance import EnergyBalanceStatus as S, estimate_fast, estimate_path
from uav_swarm_sim.planning.energy_map import EnergyMap
from uav_swarm_sim.planning.environment_map import EnvironmentMap, GridFrame
from uav_swarm_sim.planning.visibility_router import route_transit


def _plan(c, zone=None):
    return boustrophedon(zone or c.zone, c.spec, c.motion, c.em,
                         coverage=c.cfg.coverage, altitude_m=100)


def test_t9_every_path_component(energy_case):
    c = energy_case
    plan = _plan(c)
    assert plan.waypoints[0].pose.as_xy() == pytest.approx((300, 20), rel=1e-9)
    assert plan.waypoints[-1].pose.as_xy() == pytest.approx((500, 100), rel=1e-9)
    e = estimate_path(c.ctx, c.drone, c.zone, c.raster)
    turns = 2 * (240 * math.pi / 2 + 240 * 40 / 12 + 240 * math.pi / 2)
    demand = 15000 + turns + 900
    rth = 220 * 500 / 12 + 240 * 2 * math.pi + 10000
    budget = 350000 - (5500 + rth + 18000)
    expected = dict(e_level_j=360000, e_takeoff_deducted_j=10000, e_remaining_j=350000,
                    e_ferry_j=5500, e_strips_j=15000, e_connectors_j=turns,
                    e_camera_j=900, e_coverage_j=demand, e_rth_j=rth,
                    e_reserve_j=18000, budget_j=budget, demand_j=demand,
                    demand_budget_ratio=demand / budget, remaining_area_m2=24000, n_strips=3)
    for key, value in expected.items():
        assert getattr(e, key) == pytest.approx(value, rel=1e-9), key
    assert e.status is S.FEASIBLE
    assert e.method == "path" and e.drone_id == 0
    assert e.anchor_pose.as_xy() == pytest.approx((300, 20))
    assert e.exit_pose.as_xy() == pytest.approx((500, 100))


def test_t2_path_takeoff_once(energy_case):
    c = energy_case
    e = estimate_path(c.ctx, replace(c.drone, airborne=True, level_j=350000), c.zone, None)
    assert e.e_takeoff_deducted_j == pytest.approx(0)
    assert e.budget_j == pytest.approx(350000 - (5500 + 220 * 500 / 12 + 480 * math.pi + 10000 + 18000), rel=1e-9)


@pytest.mark.parametrize("deficit", [0, 2000])
def test_t5_path_nonpositive_budget(energy_case, deficit):
    c = energy_case
    # Keep the physical TURN/CRUISE/TURN summation order at the exact zero
    # boundary; regrouping this closed form changes the result by one ULP.
    return_integral = ((240 * math.pi + 220 * (500 / 12)) + 240 * math.pi) + 10000
    expenses = 5500 + return_integral + 18000
    e = estimate_path(c.ctx, replace(c.drone, airborne=True, level_j=expenses - deficit), c.zone, None)
    assert e.budget_j == pytest.approx(-deficit, abs=1e-9)
    assert e.status is S.BUDGET_NONPOSITIVE
    assert e.demand_budget_ratio is None
    assert all(math.isfinite(getattr(e, f.name)) for f in fields(e)
               if isinstance(getattr(e, f.name), float))


def test_t6b_sweep_preserves_total_but_not_individual_ferry_return(energy_case):
    """coverage_path.py:65-95 sweeps left-to-right first, then alternates.

    Reflection swaps entry/exit ends; with drone and base on x=0, total
    distance is 800 m and total yaw is 2*pi. Individual terms differ because
    their azimuths differ. Endpoint assertions must fail if the sweep changes.
    """
    c = energy_case
    mirror_zone = replace(c.zone, polygon=box(-500, 0, -300, 120),
                          entry_pose=replace(c.zone.entry_pose, heading=math.pi))
    mirror_drone = replace(c.drone, pose=replace(c.drone.pose, heading=math.pi))
    rth = RthCalculator(c.em, c.motion, c.spec, c.cfg.rth, replace(c.base, heading=math.pi), 100)
    ctx = replace(c.ctx, return_energy=lambda p, alt: rth.return_energy(p, altitude_m=alt))
    a = estimate_path(c.ctx, c.drone, c.zone, None)
    mirror = estimate_path(ctx, mirror_drone, mirror_zone, None)
    assert a.anchor_pose.as_xy() == pytest.approx((300, 20))
    assert a.exit_pose.as_xy() == pytest.approx((500, 100))
    assert mirror.anchor_pose.as_xy() == pytest.approx((-500, 20))
    assert mirror.exit_pose.as_xy() == pytest.approx((-300, 100))
    for key in ("demand_j", "budget_j", "demand_budget_ratio"):
        assert getattr(mirror, key) == pytest.approx(getattr(a, key), rel=1e-9)
    assert mirror.status is a.status
    total = 220 * 800 / 12 + 240 * 2 * math.pi + 10000
    assert a.e_ferry_j + a.e_rth_j == pytest.approx(total, rel=1e-9)
    assert mirror.e_ferry_j + mirror.e_rth_j == pytest.approx(total, rel=1e-9)
    assert mirror.e_ferry_j == pytest.approx(220 * 500 / 12 + 240 * math.pi, rel=1e-9)
    assert mirror.e_rth_j == pytest.approx(220 * 300 / 12 + 240 * math.pi + 10000, rel=1e-9)
    b = estimate_path(c.ctx, c.drone, replace(c.zone, polygon=box(700, 0, 900, 120)), None)
    b_rth = 220 * 900 / 12 + 480 * math.pi + 10000
    b_budget = 350000 - (220 * 700 / 12 + b_rth + 18000)
    assert b.e_ferry_j == pytest.approx(220 * 700 / 12, rel=1e-9)
    assert b.e_rth_j == pytest.approx(b_rth, rel=1e-9)
    assert b.budget_j == pytest.approx(b_budget, rel=1e-9)
    assert b.demand_j == pytest.approx(15000 + 1600 + 480 * math.pi + 900, rel=1e-9)
    assert b.demand_budget_ratio == pytest.approx((17500 + 480 * math.pi) / b_budget, rel=1e-9)
    assert b.demand_budget_ratio > a.demand_budget_ratio


def test_t10_raster_removes_completed_strip(energy_case):
    c = energy_case
    c.raster.record_segment(Pose(300, 20, 0), Pose(500, 20, 0), 40, 40)
    e = estimate_path(c.ctx, c.drone, c.zone, c.raster)
    assert e.remaining_area_m2 == pytest.approx(16000, rel=1e-6)
    assert e.n_strips == pytest.approx(2)
    assert e.e_strips_j == pytest.approx(250 * 400 / 10, rel=1e-9)
    assert e.e_camera_j == pytest.approx(15 * 400 / 10, rel=1e-9)
    assert e.demand_j == pytest.approx(10000 + 600 + 240 * 40 / 12 + 240 * math.pi, rel=1e-9)


@pytest.mark.parametrize("enabled", [True, False])
def test_t11_ferry_obstacle(energy_case, enabled):
    c = energy_case
    env = EnvironmentMap(box(-200, -300, 1000, 400),
                         [Obstacle(0, 0, box(75, -100, 225, 140))], 1.0)
    ctx = replace(c.ctx, env=env, coverage=replace(c.ctx.coverage, transit_free_space=enabled),
                  transit_graph_cache={})
    e = estimate_path(ctx, c.drone, c.zone, None)
    if enabled:
        independent = route_transit(c.drone.pose, Pose(300, 20, 0), c.motion, env,
                                    enabled=True, operating_area=ctx.coverage.operating_area,
                                    margin_m=ctx.coverage.operating_margin_m)
        # The existing router checks raw obstacles; env.path_clear checks the
        # buffer too. This detour grazes that buffer, so it remains blocked
        # under the estimator's explicitly stricter contract.
        assert not env.path_clear(independent)
        assert e.status is S.FERRY_BLOCKED
        assert e.demand_budget_ratio is None
        assert e.e_ferry_j == pytest.approx(c.em.path_energy(independent), rel=1e-9)
        assert e.e_ferry_j > estimate_fast(ctx, c.drone, c.zone, None).e_ferry_j
    else:
        assert e.status is S.FERRY_BLOCKED
        assert e.demand_budget_ratio is None


def test_t12_analytic_return_penalty_includes_yaw_and_landing_once(energy_case):
    c = energy_case
    env = EnvironmentMap(box(-100, -100, 1000, 400), [Obstacle(0, 0, box(200, 80, 250, 120))], 1)
    rth = RthCalculator(c.em, c.motion, c.spec, c.cfg.rth, c.base, 100, env)
    ctx = replace(c.ctx, env=env, return_energy=lambda p, a: rth.return_energy(p, altitude_m=a))
    e = estimate_path(ctx, c.drone, c.zone, None)
    assert e.e_rth_j == pytest.approx(1.5 * (220 * 500 / 12 + 480 * math.pi) + 10000, rel=1e-9)


@pytest.mark.parametrize("cost", [math.inf, math.nan, 1234.0])
def test_t12_map_exit_cell_reachability(energy_case, cost):
    c = energy_case
    frame = GridFrame(-100, -100, 100, 12, 8)
    home = np.full((12, 8), 1234.0)
    home[6, 2] = cost  # exit (500,100), independent floor((xy-origin)/100)
    emap = EnergyMap(frame, home, np.full((12, 8), -1, dtype=np.int32), np.ones((12, 8)))
    cfg = replace(c.cfg.rth, energy_map=replace(c.cfg.rth.energy_map, enabled=True, decide=True))
    rth = RthCalculator(c.em, c.motion, c.spec, cfg, c.base, 100, energy_map=emap)
    ctx = replace(c.ctx, emap=emap, return_energy=lambda p, alt: rth.return_energy(p, altitude_m=alt))
    e = estimate_path(ctx, c.drone, c.zone, None)
    if math.isfinite(cost):
        assert e.status is S.FEASIBLE
        assert e.e_rth_j == pytest.approx(1234 + 10000, rel=1e-9)
    else:
        assert e.status is S.RTH_UNREACHABLE
        assert e.demand_budget_ratio is None
        assert math.isfinite(e.e_rth_j)  # calculator's analytic fallback is retained


def test_t13_real_holonomic_strips_match_distance_integral(energy_case):
    c = energy_case
    plan = _plan(c)
    for start, end in zip(plan.waypoints[::2], plan.waypoints[1::2]):
        integral = c.em.path_energy(c.motion.plan(start.pose, end.pose, M.COVERAGE))
        assert integral == pytest.approx(250 * 200 / 10, rel=1e-9)
        assert integral == pytest.approx(c.em.distance_energy(200, M.COVERAGE, 10), rel=1e-9)


def test_t14_derived_fast_path_bounds(energy_case):
    c = energy_case
    fast = estimate_fast(c.ctx, c.drone, c.zone, None)
    path = estimate_path(c.ctx, c.drone, c.zone, None)
    assert fast.e_strips_j == pytest.approx(path.e_strips_j, rel=1e-9)
    assert fast.e_camera_j == pytest.approx(path.e_camera_j, rel=1e-9)
    assert abs(fast.e_coverage_j - path.e_coverage_j) <= 0.10 * path.e_coverage_j
    half_diagonal_energy = 220 * (math.hypot(200, 120) / 2) / 12
    assert abs(fast.e_ferry_j - path.e_ferry_j) <= half_diagonal_energy
    assert abs(fast.e_rth_j - path.e_rth_j) <= half_diagonal_energy


@pytest.mark.parametrize("height", [10, 120])
def test_t15_multipolygon_components_sum_without_changing_total(energy_case, height):
    c = energy_case
    # Same separated rectangles as test_coverage_path_multipolygon.py, plus full strips.
    geometry = MultiPolygon([box(0, 0, 100, height), box(200, 0, 300, height)])
    zone = replace(c.zone, polygon=geometry)
    plan = boustrophedon(zone, c.spec, c.motion, c.em,
                         env=EnvironmentMap(geometry, [], 0), coverage=c.ctx.coverage, altitude_m=100)
    assert plan.strips_energy_j + plan.connectors_energy_j == pytest.approx(plan.est_energy_j, abs=1e-6)
    # At height 120 the long axis is vertical: six 120 m strips, not 100 m.
    assert plan.strips_energy_j == pytest.approx(250 * (200 if height == 10 else 720) / 10, rel=1e-9)


def test_empty_path_and_photogrammetry_guard(energy_case):
    c = energy_case
    e = estimate_path(c.ctx, c.drone, replace(c.zone, polygon=Polygon()), None)
    assert e.demand_j == pytest.approx(0)
    assert e.demand_budget_ratio == pytest.approx(0)
    assert e.anchor_pose == e.exit_pose == c.zone.entry_pose
    with pytest.raises(AssertionError, match="requires photogrammetry"):
        estimate_path(replace(c.ctx, spec=replace(c.spec, photogrammetry=None)), c.drone, c.zone, None)


@pytest.mark.parametrize("estimator", [estimate_fast, estimate_path])
def test_optional_environment_with_transit_enabled(energy_case, estimator):
    c = energy_case
    ctx = replace(c.ctx, coverage=replace(c.ctx.coverage, transit_free_space=True))
    expected = estimator(c.ctx, c.drone, c.zone, None)
    actual = estimator(ctx, c.drone, c.zone, None)
    assert actual.status is S.FEASIBLE
    assert actual.e_ferry_j == pytest.approx(expected.e_ferry_j, rel=1e-9)
