"""EXP-09 physical fixtures: M4E power, test-commanded 10 m/s and 100 m AGL."""
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
from shapely.geometry import box

from uav_swarm_sim.execution.agent import Agent
from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.execution.state_machine import AgentContext, StateMachine, Transition
from uav_swarm_sim.infrastructure.config import load_config, ConfigError
from uav_swarm_sim.infrastructure.core_types import Path, PathSegment, Pose, CoveragePlan, Waypoint
from uav_swarm_sim.infrastructure.enums import AgentState as S, BatteryZone, ManeuverType as M
from uav_swarm_sim.physical_model.battery import Battery
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.physical_model.vertical_segments import landing_profile
from uav_swarm_sim.planning.energy_map import build_energy_map
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.obstacle_generator import Obstacle
from uav_swarm_sim.planning.visibility_router import RouteUnavailable, route_transit, route_connector, _path_clear


@pytest.fixture
def kit():
    cfg = load_config("config/djimatrice4e.yaml", overrides={
        "platforms.MULTIROTOR.v_cruise": 10.0,
        "platforms.MULTIROTOR.v_coverage": 10.0,
        "env.coverage_altitude_m": 100.0,
        "sensor.photogrammetry.enabled": True,
        "sensor.sensor_power_w": 20.0,  # ASSUMPTION for tests, not a DJI specification
        "coverage.raster_enabled": True,
        "coverage.transit_free_space": True, "coverage.ferry_free_space": True,
        "mission.no_swap_mode": True, "rth.execution_coherent": True,
        "rth.emergency_frac": 0.05, "rth.energy_map.zone_demotion": True,
    })
    spec = build_spec(cfg)
    return SimpleNamespace(cfg=cfg, spec=spec, motion=make_motion_model(spec), em=EnergyModel(spec))


def environment(blocked=False):
    obs = [Obstacle(0, 0, box(450, 200, 550, 800))] if blocked else []
    return EnvironmentMap(box(-100, -100, 1200, 1100), obs, 5.0)


def calculator(k, env=None, map_on=False):
    env = env or environment()
    cfg = k.cfg.rth
    base = Pose(0, 500, 0)
    emap = None
    if map_on:
        cfg = replace(cfg, energy_map=replace(cfg.energy_map, enabled=True, route=True, decide=True))
        emap = build_energy_map(env, base, 20, k.em, 10)
    return RthCalculator(k.em, k.motion, k.spec, cfg, base, 100, env,
                         energy_map=emap, coverage=k.cfg.coverage)


def agent(k, rth, base=Pose(0, 500, 0)):
    a = Agent(0, k.spec, k.motion, k.em, Battery(k.spec.battery_capacity_j, k.cfg.battery_zones),
              StateMachine(k.cfg.battery_zones, zone_demotion=True, no_swap_mode=True), rth,
              None, base, coverage_altitude_m=100, sensor_power_w=20, photo_spacing_m=20)
    return a


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("kind", ["transit", "connector", "return"])
def test_straight_and_detour_use_buffered_geometry(kit, kind, blocked):
    env = environment(blocked)
    start, end = Pose(0, 500, 0), Pose(1000, 500, 0)
    if kind == "return":
        path = calculator(kit, env).return_plan(end, base=start).path
    else:
        router = route_transit if kind == "transit" else route_connector
        path = router(start, end, kit.motion, env, enabled=True)
    assert _path_clear(path, env)
    assert path.total_length_m > 1000 if blocked else path.total_length_m == pytest.approx(1000)
    for seg in path.segments:
        if seg.length_m:
            assert seg.speed == pytest.approx(10)
        elif seg.maneuver is M.TURN:
            assert seg.speed == 0


@pytest.mark.parametrize("map_on", [False, True])
@pytest.mark.parametrize("dt", [0.1, 0.5, 1.3])
def test_return_prediction_equals_execution_yaw_and_100m_landing(kit, map_on, dt):
    rth = calculator(kit, environment(True), map_on)
    a = agent(kit, rth)
    a.pose = Pose(1000, 500, math.pi / 2)
    a.state = S.S2_MISSION
    a._cov_legs = a._legs = [kit.motion.plan(Pose(0, 500, 0), Pose(100, 500, 0), M.COVERAGE)]
    a._coherent.altitude_m = 100
    expected = rth.return_plan(a.pose, base=a.base, altitude_m=100)
    events = []
    bus = SimpleNamespace(publish=events.append)
    a._apply_transition(Transition(a.state, S.S3_RTH, "rth_energy"), 0, bus)
    for i in range(30000):
        a.step(dt, i * dt, bus)
        if a.state is S.S_LANDED:
            break
    assert a.state is S.S_LANDED
    assert a._coherent.altitude_m == pytest.approx(0, abs=1e-8)
    assert a.energy_consumed_j == pytest.approx(expected.energy_j, rel=1e-10, abs=1e-6)
    assert a.pose.as_xy() == pytest.approx(a.base.as_xy())
    assert not a.photo_events
    assert not a.rth_infeasible_events


def test_independent_budget_example_and_single_reserve(kit):
    rth = calculator(kit)
    start = Pose(1000, 500, 0)
    base = Pose(0, 500, math.pi)
    expected_return = 156.3 * 100 + 149.2 * math.pi + 106.6 * 12.5
    # West-facing home requires one pi rotation, matching the approved example.
    assert rth.return_energy(start, base=base) == pytest.approx(expected_return)
    required = 1692 + expected_return + 17910
    assert not rth.should_return(required + 1, 1692, start, base=base)
    assert rth.should_return(required - 1, 1692, start, base=base)
    assert rth.reserve_j == 17910


@pytest.mark.parametrize("change", ["geometry", "actual_pose"])
def test_changed_executed_return_replans_and_refreshes_energy(kit, change):
    rth = calculator(kit)
    a = agent(kit, rth)
    a.pose = Pose(1000, 500, 0)
    a.state = S.S2_MISSION
    a._coherent.altitude_m = 100
    bus = SimpleNamespace(publish=lambda event: None)
    a._apply_transition(Transition(a.state, S.S3_RTH, "rth_energy"), 0, bus)
    previous = a._coherent.return_plan
    if change == "geometry":
        rth._env = environment(True)
    else:
        a.pose = Pose(1000, 600, math.pi / 2)
    expected = rth.return_plan(a.pose, base=a.base, altitude_m=100)
    assert expected.energy_j != previous.energy_j
    for i in range(3000):
        a.step(0.5, i * 0.5, bus)
        if a.state is S.S_LANDED:
            break
    assert a.state is S.S_LANDED
    assert a._coherent.return_plan == expected
    assert a.energy_consumed_j == pytest.approx(expected.energy_j, rel=1e-10, abs=1e-6)


def test_reporting_bins_and_emergency_guard_are_independent(kit):
    assert kit.cfg.battery_zones.critical == 0.20
    a = agent(kit, calculator(kit))
    a.battery.drain(0.9 * a.battery.capacity_j)
    a.state = S.S2_MISSION
    a._cov_legs = a._legs = [kit.motion.plan(Pose(0, 500, 0), Pose(100, 500, 0), M.COVERAGE)]
    assert a.battery.zone is BatteryZone.TERMINAL
    assert not a._make_ctx().emergency_battery
    assert a.sm.step(a._make_ctx()) is None
    a.battery.drain(0.06 * a.battery.capacity_j)
    assert a.sm.step(a._make_ctx()).reason == "terminal_battery"
    legacy = AgentContext(S.S2_MISSION, BatteryZone.TERMINAL)
    assert a.sm.step(legacy).reason == "terminal_battery"


def test_config_map_not_required_and_invalid_emergency_rejected(kit):
    assert not kit.cfg.rth.energy_map.enabled
    for value in (-0.1, 1, float("nan"), float("inf")):
        with pytest.raises(ConfigError, match="emergency_frac"):
            load_config("config/djimatrice4e.yaml", overrides={"rth.emergency_frac": value})
    assert load_config("config/djimatrice4e.yaml").rth.emergency_frac is None


@pytest.mark.parametrize("router", [route_transit, route_connector])
def test_no_route_and_invalid_fallback_are_rejected(kit, router, monkeypatch):
    import uav_swarm_sim.planning.visibility_router as routing
    env = environment(True)
    start, end = Pose(0, 500, 0), Pose(1000, 500, 0)
    monkeypatch.setattr(routing, "_shortest_polyline", lambda *args: None)
    with pytest.raises(RouteUnavailable):
        router(start, end, kit.motion, env, enabled=True)
    # A lying/invalid router candidate is checked again, not trusted.
    monkeypatch.setattr(routing, "_shortest_polyline", lambda *args: [start.as_xy(), end.as_xy()])
    with pytest.raises(RouteUnavailable):
        router(start, end, kit.motion, env, enabled=True)


def test_stale_map_uses_current_visibility_geometry(kit):
    rth = calculator(kit, environment(), True)
    rth._env = environment(True)
    plan = rth.return_plan(Pose(1000, 500, 0))
    assert plan.path.total_length_m > 1000
    assert _path_clear(plan.path, rth._env)


def test_unreachable_return_hovers_and_remains_falsifiable(kit):
    env = EnvironmentMap(box(-100, -100, 1200, 1100),
                         [Obstacle(0, 0, box(450, -200, 550, 1200))], 5)
    a = agent(kit, calculator(kit, env))
    a.state, a.pose = S.S2_MISSION, Pose(1000, 500, 0)
    a._coherent.altitude_m = 100
    bus = SimpleNamespace(publish=lambda event: None)
    a._apply_transition(Transition(a.state, S.S3_RTH, "rth_energy"), 0, bus)
    assert a._coherent.holding
    assert a.rth_infeasible_events[0].payload["deficit_j"] is None
    a.battery._level = 100
    a.step(1, 1, bus)
    assert a.battery.level_j == 0
    assert a.state is S.S3_RTH  # the engine, not a fabricated touchdown, kills it
    assert a.pose.as_xy() == (1000, 500)


def test_old_tick_bias_has_a_measured_positive_sign(kit):
    # 101 m / 8 m/s = 12.625 s; dt=.5 used to bill 13 s.
    land = landing_profile(kit.spec, kit.em, 101).as_path()
    old = 0.0
    elapsed = 0.0
    while elapsed < land.total_duration_s - 1e-9:
        old += kit.em.segment_energy(land.maneuver_at_time(elapsed), 0.5)
        elapsed = min(elapsed + 0.5, land.total_duration_s)
    exact = 106.6 * 101 / 8
    assert old - exact == pytest.approx(39.975)
    assert kit.em.path_interval_energy(land, 0, land.total_duration_s) == pytest.approx(exact)
    # The requested 100 m/.5 s case happens to be aligned: zero tail bias.
    assert math.ceil((100 / 8) / 0.5) * 0.5 * 106.6 == pytest.approx(1332.5)


def test_multirotor_takeoff_carries_no_potential_term(kit):
    """Scope check for the clamp below: multirotor TAKEOFF/LAND carry the
    altitude in the DURATION at dz=0, and coherent mode is validated to a
    single layer, so no potential term arises in a supported coherent flight.
    The m*g*dz term belongs to the inter-layer CLIMB segments."""
    from uav_swarm_sim.physical_model.vertical_segments import takeoff_profile
    seg = takeoff_profile(kit.spec, kit.em, 100, at=Pose(0, 500, 0)).as_path().segments[0]
    assert seg.start.z == seg.end.z == 0.0
    assert kit.em.potential_power_w(seg) == 0.0


def test_depletion_clamp_never_reports_more_energy_than_the_battery_held(kit):
    """The truncation must use the SAME integral that charges the segment.

    path_interval_energy bills mass*g*dz/duration on a climbing segment.
    Clamping on propulsion power alone cuts at a point the battery cannot pay
    for: Battery.drain floors the level at 0 but energy_consumed_j does not,
    so the reported total would exceed the energy the battery supplied. No
    such segment occurs in the currently validated single-layer scope; this
    pins the arithmetic so that widening the scope cannot break the books."""
    a = agent(kit, calculator(kit))
    climb = Path((PathSegment(M.CLIMB, 0.0, 10.0,
                              Pose(0, 500, 0, 0.0), Pose(0, 500, 0, 50.0), 0.0),))
    seg = climb.segments[0]
    rate = kit.em.power(M.CLIMB) + kit.em.potential_power_w(seg)
    assert kit.em.potential_power_w(seg) > 0
    a.state = S.S1_TRANSIT
    a._set_legs([climb])
    held = 0.5 * rate  # exactly half a second of climb left
    a.battery.drain(a.battery.level_j - held)
    a._coherent.tick(5.0, 0.0)
    assert a.battery.level_j == 0
    assert a.energy_consumed_j == pytest.approx(held, rel=1e-9)


def work_agent(k, rth):
    a = agent(k, rth)
    poses = [Pose(100, 500, 0), Pose(200, 500, 0),
             Pose(200, 550, math.pi), Pose(100, 550, math.pi)]
    plan = CoveragePlan(0, [Waypoint(p, M.COVERAGE, 10) for p in poses], 0, 0)
    a.assign(plan, k.motion.plan(a.base, poses[0], M.CRUISE))
    a.state, a.pose = S.S2_MISSION, poses[0]
    a._set_legs(a._cov_legs)
    a._coherent.altitude_m = 100
    return a


@pytest.mark.parametrize("interval", [0.1, 5.0, 1000.0])
@pytest.mark.parametrize("dt", [0.1, 0.5, 30.0])
def test_bundle_admission_precedes_movement_even_with_long_dt(kit, interval, dt):
    rth = calculator(kit)
    rth._cfg = replace(rth._cfg, check_interval_s=interval)
    for margin in (-1, 1):
        a = work_agent(kit, rth)
        next_cost, end = a._coherent.bundle()
        required = next_cost + rth.return_energy(end, base=a.base, altitude_m=100) + rth.reserve_j
        a.battery._level = required + margin
        bus = SimpleNamespace(publish=lambda event: None)
        a.step(dt, 0, bus)
        if margin < 0:
            assert a.state in (S.S3_RTH, S.S_LANDED)
            assert not a.photo_events
        else:
            assert a.photo_events
        for i in range(2000):
            if a.state is S.S_LANDED:
                break
            a.step(dt, (i + 1) * dt, bus)
        assert a.state is S.S_LANDED
        assert a.battery.level_j >= rth.reserve_j - 1e-6


def test_partial_strip_lookahead_and_camera_only_on_productive_motion(kit):
    a = work_agent(kit, calculator(kit))
    a._t = 4
    a.pose = a._legs[0].pose_at_time(4)
    cost, end = a._coherent.bundle()
    assert cost == pytest.approx((149.2 + 20) * 6)
    assert end.as_xy() == (200, 500)
    connector = a._cov_legs[1]
    assert a._coherent.energy(connector, camera=False) == pytest.approx(
        149.2 * (50 / 10 + math.pi))


def test_per_drone_base_is_used_for_both_route_and_energy(kit):
    rth = calculator(kit)
    source = Pose(100, 500, math.pi)
    near = Pose(10, 500, math.pi)
    far = Pose(0, 500, math.pi)
    p = rth.return_plan(source, base=near)
    assert p.path.end_pose == near
    assert rth.return_energy(source, base=far) - p.energy_j == pytest.approx(156.3)


def test_buffer_intrusion_is_rejected_without_changing_safety_predicate(kit):
    env = EnvironmentMap(box(0, 0, 100, 100), [Obstacle(0, 0, box(40, 40, 60, 60))], 5)
    start, end = Pose(10, 63, 0), Pose(90, 63, 0)
    chord = kit.motion.plan(start, end, M.CRUISE)
    assert not env.segment_in_obstacle(start, end)  # no hard penetration
    assert not _path_clear(chord, env)  # soft buffer: planner rejects it
    path = route_transit(start, end, kit.motion, env, enabled=True)
    assert _path_clear(path, env)


def test_endpoint_in_obstacle_is_not_a_valid_empty_route(kit):
    env = environment(True)
    inside = Pose(500, 500, 0)
    with pytest.raises(RouteUnavailable):
        calculator(kit, env).return_plan(inside, base=inside)


def test_known_energy_deficit_is_diagnostic_and_depletion_is_not_landing(kit):
    rth = calculator(kit)
    a = agent(kit, rth)
    a.state, a.pose = S.S2_MISSION, Pose(1000, 500, 0)
    a._coherent.altitude_m = 100
    required = rth.return_energy(a.pose, base=a.base, altitude_m=100)
    a.battery._level = 100
    bus = SimpleNamespace(publish=lambda event: None)
    a._apply_transition(Transition(a.state, S.S3_RTH, "rth_energy"), 2, bus)
    assert a.rth_infeasible_events[0].payload == {
        "agent_id": 0, "deficit_j": pytest.approx(required - 100),
        "reason": "insufficient_return_energy",
    }
    a.step(5, 2, bus)
    assert a.state is S.S3_RTH
    assert a.battery.level_j == 0
    assert a.energy_consumed_j == pytest.approx(100)
    assert a._coherent.altitude_m == 100


def test_emergency_override_changes_return_without_changing_reporting_bins(kit):
    low = work_agent(kit, calculator(kit))
    high_calc = calculator(kit)
    high_calc._cfg = replace(high_calc._cfg, emergency_frac=0.95)
    high = work_agent(kit, high_calc)
    for a in (low, high):
        a.battery._level = 0.90 * a.battery.capacity_j
        assert a.battery.zone is BatteryZone.HIGH
        a.step(0.5, 0, SimpleNamespace(publish=lambda event: None))
    assert low.state is S.S2_MISSION
    assert high.state is S.S3_RTH


def test_exp06_path_contract_uses_real_return_even_when_grid_says_infinite(kit):
    from uav_swarm_sim.infrastructure.core_types import Zone
    from uav_swarm_sim.planning.energy_balance import (
        build_energy_balance_context, DroneEnergyState, estimate_path, EnergyBalanceStatus,
    )
    env = environment()
    rth = calculator(kit, env, True)
    rth._map.e_home[:] = float("inf")  # grid heuristic cannot override visibility reachability
    base = Pose(10, 500, 0)
    ctx = build_energy_balance_context(
        kit.cfg, kit.em, kit.spec, kit.motion, env,
        lambda pose, alt: rth.return_energy(pose, altitude_m=alt, base=base), emap=rth._map,
    )
    state = DroneEnergyState(0, base, 358200, False)
    zone = Zone(0, [], box(100, 450, 300, 550), Pose(100, 450, 0))
    estimate = estimate_path(ctx, state, zone, None)
    assert estimate.status is EnergyBalanceStatus.FEASIBLE
    assert estimate.e_rth_j == pytest.approx(rth.return_energy(estimate.exit_pose, base=base))
    assert estimate.budget_j == pytest.approx(
        358200 - 1989 - estimate.e_ferry_j - estimate.e_rth_j - 17910)
    assert estimate.demand_j == pytest.approx(
        estimate.e_strips_j + estimate.e_connectors_j + estimate.e_camera_j)


def test_complete_flight_cache_byte_identity(kit):
    signatures = []
    for cache in (None, {}):
        env = environment(True)
        rth = calculator(kit, env)
        a = agent(kit, rth)
        points = [Pose(1000, 500, 0), Pose(1100, 500, 0)]
        plan = CoveragePlan(0, [Waypoint(p, M.COVERAGE, 10) for p in points], 100, 1492)
        transit = route_transit(a.base, points[0], kit.motion, env, enabled=True, graph_cache=cache)
        assert transit.total_length_m > 1000
        if cache is not None:
            assert cache  # compare the cached branch non-vacuously
        a.assign(plan, transit)
        trace, events = [], []
        bus = SimpleNamespace(publish=events.append)
        for i in range(3000):
            a.step(0.5, i * 0.5, bus)
            trace.append((a.state, a.pose, a.battery.level_j))
            if a.state is S.S_LANDED:
                break
        assert a.state is S.S_LANDED
        signatures.append((trace, events, a.photo_events, a.energy_consumed_j))
    assert signatures[0] == signatures[1]
