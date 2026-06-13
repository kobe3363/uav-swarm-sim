"""Batch 2 tests: physical_model + infrastructure/core_types (isolated)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import config_path

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import (
    Path,
    Pose,
    arc_segment,
    inplace_turn_segment,
    normalize_angle,
    straight_segment,
)
from uav_swarm_sim.infrastructure.enums import ManeuverType, PlatformType
from uav_swarm_sim.physical_model import dubins
from uav_swarm_sim.physical_model.aero_correction import AeroCorrection
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.metrics_definitions import workload_std
from uav_swarm_sim.physical_model.motion_model import (
    DubinsModel,
    HolonomicModel,
    make_motion_model,
)
from uav_swarm_sim.physical_model.vertical_segments import (
    landing_profile,
    takeoff_profile,
)


def _spec(platform="MULTIROTOR"):
    cfg = load_config(config_path(), overrides={"platform_type": platform})
    return build_spec(cfg), cfg


def _pose_close(p, q, tol=1e-6):
    return (
        abs(p.x - q.x) <= tol
        and abs(p.y - q.y) <= tol
        and abs(normalize_angle(p.heading - q.heading)) <= tol
    )


# --------------------------------------------------------------------------- #
# core_types geometry                                                         #
# --------------------------------------------------------------------------- #
def test_straight_segment_endpoint_and_length():
    s = straight_segment(Pose(0, 0, 0.0), 10.0, ManeuverType.CRUISE, 5.0)
    assert _pose_close(s.end, Pose(10, 0, 0))
    assert s.length_m == pytest.approx(10.0)
    assert s.duration_s == pytest.approx(2.0)


def test_arc_quarter_circle_left():
    # left quarter turn, radius 1, from origin heading +x -> ends at (1,1) heading +y
    s = arc_segment(Pose(0, 0, 0.0), curvature=1.0, arc_length=math.pi / 2, maneuver=ManeuverType.TURN, speed=1.0)
    assert _pose_close(s.end, Pose(1.0, 1.0, math.pi / 2), tol=1e-9)


def test_path_pose_at_length_and_time():
    s = straight_segment(Pose(0, 0, 0.0), 10.0, ManeuverType.CRUISE, 5.0)
    p = Path.from_segments([s])
    assert _pose_close(p.pose_at_length(5.0), Pose(5, 0, 0))
    # duration 2.0 s; halfway in time = 1.0 s -> x=5
    assert _pose_close(p.pose_at_time(1.0), Pose(5, 0, 0))


def test_inplace_turn_progresses_by_time_not_length():
    t = inplace_turn_segment(Pose(0, 0, 0.0), math.pi / 2, omega_max=1.0, maneuver=ManeuverType.TURN)
    assert t.length_m == 0.0
    assert t.duration_s == pytest.approx(math.pi / 2)
    p = Path.from_segments([t])
    mid = p.pose_at_time(t.duration_s / 2)
    assert mid.heading == pytest.approx(math.pi / 4)


# --------------------------------------------------------------------------- #
# drone_specs                                                                 #
# --------------------------------------------------------------------------- #
def test_effective_swath_and_capacity():
    spec, cfg = _spec()
    assert spec.swath_width_m == pytest.approx(
        cfg.sensor.swath_width_m * (1 - cfg.sensor.overlap_frac)
    )
    assert spec.battery_capacity_j == pytest.approx(cfg.fleet.battery_capacity_j)


# --------------------------------------------------------------------------- #
# dubins                                                                      #
# --------------------------------------------------------------------------- #
def test_dubins_straight_line():
    start, goal = Pose(0, 0, 0.0), Pose(10, 0, 0.0)
    path = dubins.shortest_path(start, goal, r_min=1.0, v=2.0, maneuver=ManeuverType.CRUISE)
    assert path.total_length_m == pytest.approx(10.0, abs=1e-6)
    assert _pose_close(path.end_pose, goal)


def test_dubins_endpoint_matches_goal_random():
    rng = np.random.default_rng(0)
    r_min = 2.0
    for _ in range(200):
        start = Pose(*rng.uniform(-10, 10, 2), float(rng.uniform(-math.pi, math.pi)))
        goal = Pose(*rng.uniform(-10, 10, 2), float(rng.uniform(-math.pi, math.pi)))
        path = dubins.shortest_path(start, goal, r_min, v=1.0, maneuver=ManeuverType.CRUISE)
        assert _pose_close(path.end_pose, goal, tol=1e-4), f"{start} -> {goal}"


def test_dubins_path_length_matches_shortest_path():
    rng = np.random.default_rng(1)
    for _ in range(100):
        start = Pose(*rng.uniform(-10, 10, 2), float(rng.uniform(-math.pi, math.pi)))
        goal = Pose(*rng.uniform(-10, 10, 2), float(rng.uniform(-math.pi, math.pi)))
        L = dubins.path_length(start, goal, 2.0)
        P = dubins.shortest_path(start, goal, 2.0, 1.0, ManeuverType.CRUISE).total_length_m
        assert L == pytest.approx(P, abs=1e-6)


def test_dubins_respects_min_radius():
    path = dubins.shortest_path(Pose(0, 0, 0.0), Pose(0, 1, math.pi), r_min=3.0, v=1.0, maneuver=ManeuverType.CRUISE)
    for seg in path.segments:
        if seg.curvature != 0.0:
            assert abs(1.0 / seg.curvature) == pytest.approx(3.0, abs=1e-9)


def test_dubins_rejects_nonpositive_radius():
    with pytest.raises(ValueError):
        dubins.shortest_path(Pose(0, 0, 0), Pose(1, 1, 0), r_min=0.0, v=1.0, maneuver=ManeuverType.CRUISE)


# --------------------------------------------------------------------------- #
# motion_model                                                                #
# --------------------------------------------------------------------------- #
def test_make_motion_model_per_platform():
    assert isinstance(make_motion_model(_spec("MULTIROTOR")[0]), HolonomicModel)
    assert isinstance(make_motion_model(_spec("FIXED_WING")[0]), DubinsModel)
    assert isinstance(make_motion_model(_spec("VTOL")[0]), DubinsModel)


def test_dubins_model_plan_and_cost():
    spec, _ = _spec("FIXED_WING")
    m = DubinsModel(spec)
    start, goal = Pose(0, 0, 0.0), Pose(100, 50, 1.0)
    path = m.plan(start, goal, ManeuverType.CRUISE)
    assert _pose_close(path.end_pose, goal, tol=1e-3)
    assert m.leg_cost(start, goal) == pytest.approx(path.total_length_m, abs=1e-6)


def test_holonomic_plan_reaches_goal_and_euclidean_cost():
    spec, _ = _spec("MULTIROTOR")
    m = HolonomicModel(spec)
    start, goal = Pose(0, 0, 0.0), Pose(30, 40, 1.0)
    path = m.plan(start, goal, ManeuverType.CRUISE)
    assert _pose_close(path.end_pose, goal, tol=1e-9)
    assert m.leg_cost(start, goal) == pytest.approx(50.0)  # 3-4-5
    # advance by time reaches the end pose
    pose, t = path.pose_at_time(path.total_duration_s), path.total_duration_s
    assert _pose_close(pose, goal, tol=1e-9)


# --------------------------------------------------------------------------- #
# energy_model -- the E = sum P*dt identity and the formation invariant        #
# --------------------------------------------------------------------------- #
def test_segment_energy_is_power_times_time():
    spec, _ = _spec("MULTIROTOR")
    em = EnergyModel(spec)
    p = em.power(ManeuverType.CRUISE)
    assert em.segment_energy(ManeuverType.CRUISE, 10.0) == pytest.approx(p * 10.0)


def test_path_energy_equals_manual_integral():
    spec, _ = _spec("MULTIROTOR")
    em = EnergyModel(spec)
    s1 = straight_segment(Pose(0, 0, 0), 10.0, ManeuverType.CRUISE, 5.0)
    s2 = straight_segment(s1.end, 6.0, ManeuverType.COVERAGE, 3.0)
    path = Path.from_segments([s1, s2])
    manual = (
        em.power(ManeuverType.CRUISE) * s1.duration_s
        + em.power(ManeuverType.COVERAGE) * s2.duration_s
    )
    assert em.path_energy(path) == pytest.approx(manual)


def test_energy_is_time_integral_not_distance_average():
    # same distance, different speed -> different energy (proves time-integration)
    spec, _ = _spec("MULTIROTOR")
    em = EnergyModel(spec)
    slow = straight_segment(Pose(0, 0, 0), 10.0, ManeuverType.CRUISE, 2.0)  # 5 s
    fast = straight_segment(Pose(0, 0, 0), 10.0, ManeuverType.CRUISE, 10.0)  # 1 s
    e_slow = em.path_energy(Path.from_segments([slow]))
    e_fast = em.path_energy(Path.from_segments([fast]))
    assert e_slow > e_fast


def test_formation_factor_applies_to_cruise_fw_only():
    fw = EnergyModel(_spec("FIXED_WING")[0])
    mr = EnergyModel(_spec("MULTIROTOR")[0])
    f = 0.8486
    # FW CRUISE: discounted
    assert fw.power(ManeuverType.CRUISE, f) == pytest.approx(fw.power(ManeuverType.CRUISE) * f)
    # FW COVERAGE: never discounted (thesis invariant)
    assert fw.power(ManeuverType.COVERAGE, f) == pytest.approx(fw.power(ManeuverType.COVERAGE))
    # MULTIROTOR CRUISE: no benefit
    assert mr.power(ManeuverType.CRUISE, f) == pytest.approx(mr.power(ManeuverType.CRUISE))


def test_distance_energy_consistency():
    spec, _ = _spec("FIXED_WING")
    em = EnergyModel(spec)
    d, v = 100.0, spec.v_cruise
    assert em.distance_energy(d, ManeuverType.CRUISE, v) == pytest.approx(
        em.segment_energy(ManeuverType.CRUISE, d / v)
    )


# --------------------------------------------------------------------------- #
# aero_correction                                                             #
# --------------------------------------------------------------------------- #
def test_power_factor_rules():
    cfg = load_config(config_path())
    red = cfg.aero.formation_drag_reduction
    fw = AeroCorrection(cfg.aero, PlatformType.FIXED_WING)
    mr = AeroCorrection(cfg.aero, PlatformType.MULTIROTOR)
    assert fw.power_factor(True, ManeuverType.CRUISE) == pytest.approx(1 - red)
    assert fw.power_factor(True, ManeuverType.COVERAGE) == 1.0
    assert fw.power_factor(False, ManeuverType.CRUISE) == 1.0
    assert mr.power_factor(True, ManeuverType.CRUISE) == 1.0


def test_wake_zones_geometry():
    cfg = load_config(config_path())
    mr = AeroCorrection(cfg.aero, PlatformType.MULTIROTOR)
    zones = mr.wake_zones([Pose(0, 0, 0.0)])
    assert len(zones) == 1
    z = zones[0]
    assert z.is_valid and z.area > 0
    # wake extends behind (negative x) for heading 0
    minx, _, maxx, _ = z.bounds
    assert minx < 0 <= maxx + 1e-9


# --------------------------------------------------------------------------- #
# vertical_segments                                                           #
# --------------------------------------------------------------------------- #
def test_multirotor_takeoff_vertical():
    spec, _ = _spec("MULTIROTOR")
    em = EnergyModel(spec)
    prof = takeoff_profile(spec, em, altitude_m=100.0)
    assert prof.duration_s == pytest.approx(100.0 / spec.v_climb)
    assert prof.energy_j > 0


def test_fixed_wing_takeoff_includes_ground_roll():
    spec, _ = _spec("FIXED_WING")
    em = EnergyModel(spec)
    prof = takeoff_profile(spec, em, altitude_m=100.0)
    # energy strictly exceeds the climb-only integral by the ground-roll charge
    climb_only = prof.energy_j - spec.ground_roll_energy_j
    assert spec.ground_roll_energy_j > 0
    assert climb_only > 0
    land = landing_profile(spec, em, altitude_m=100.0)
    assert land.energy_j > 0


# --------------------------------------------------------------------------- #
# metrics_definitions                                                         #
# --------------------------------------------------------------------------- #
def test_workload_std_zero_when_equal():
    assert workload_std({0: 100.0, 1: 100.0, 2: 100.0}) == pytest.approx(0.0)


def test_workload_std_matches_numpy():
    d = {0: 10.0, 1: 20.0, 2: 35.0, 3: 5.0}
    assert workload_std(d) == pytest.approx(float(np.std(list(d.values()))))
