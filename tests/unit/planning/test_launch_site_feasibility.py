from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.infrastructure.enums import ManeuverType, PlatformType
from uav_swarm_sim.physical_model.aero_correction import AeroCorrection
from uav_swarm_sim.physical_model.drone_specs import PlatformSpec, build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.gvg_builder import build_gvg
from uav_swarm_sim.planning.launch_site_optimizer import (
    InfeasibleMissionError,
    furthest_point_feasible,
    optimize,
    _furthest_free_vertex_dist,
)
from uav_swarm_sim.planning.tgc import build_tgc

CONFIG_PATH = "config/default.yaml"
ALT = 100.0


# --------------------------------------------------------------------------- #
# hand-built platform so the pure tests need no config / builders             #
# --------------------------------------------------------------------------- #
def _multirotor_spec(battery_j: float) -> PlatformSpec:
    """A ~2 kg quad matching the default.yaml MULTIROTOR table, with an
    overridable battery so feasibility can be driven across its threshold."""
    return PlatformSpec(
        platform=PlatformType.MULTIROTOR,
        v_cruise=12.0,
        v_coverage=6.0,
        v_climb=3.0,
        v_descent=2.5,
        r_min_m=0.0,
        omega_max=1.0,
        power_w={
            ManeuverType.IDLE: 12.0,
            ManeuverType.TAKEOFF: 280.0,
            ManeuverType.CLIMB: 280.0,
            ManeuverType.CRUISE: 220.0,
            ManeuverType.COVERAGE: 210.0,
            ManeuverType.TURN: 210.0,
            ManeuverType.DESCENT: 140.0,
            ManeuverType.LAND: 150.0,
            ManeuverType.HOVER: 200.0,
        },
        battery_capacity_j=battery_j,
        dims_m=(1.2, 1.2, 0.4),
        swath_width_m=50.0,
        climb_angle_rad=math.pi / 2,
        ground_roll_energy_j=0.0,
        mass_kg=2.0,
    )


def _vertical_j(em: EnergyModel, spec: PlatformSpec) -> float:
    # MULTIROTOR vertical phases are table-only (no mass term; dz==0 segment).
    take = spec.power_w[ManeuverType.TAKEOFF] * (ALT / spec.v_climb)
    land = spec.power_w[ManeuverType.LAND] * (ALT / spec.v_descent)
    return take + land


# --------------------------------------------------------------------------- #
# pure: furthest_point_feasible                                               #
# --------------------------------------------------------------------------- #
def test_feasible_when_point_is_well_within_range():
    spec = _multirotor_spec(360_000.0)  # 100 Wh
    em = EnergyModel(spec)
    # 1 km away: ~52 kJ round trip + vertical, far under 342 kJ usable
    assert furthest_point_feasible(em, spec, 1_000.0, ALT) is True


def test_infeasible_when_point_is_too_far():
    spec = _multirotor_spec(360_000.0)
    em = EnergyModel(spec)
    # 10 km away: ~382 kJ needed > 342 kJ usable
    assert furthest_point_feasible(em, spec, 10_000.0, ALT) is False


def test_feasibility_boundary_is_inclusive():
    spec = _multirotor_spec(360_000.0)
    em = EnergyModel(spec)
    usable = spec.battery_capacity_j * 0.95
    rate = spec.power_w[ManeuverType.CRUISE] / spec.v_cruise  # J per metre, cruise
    # solve 2*d*rate + vertical == usable  -> d at the exact threshold
    d_exact = (usable - _vertical_j(em, spec)) / (2.0 * rate)
    assert furthest_point_feasible(em, spec, d_exact, ALT) is True          # '<=' is inclusive
    assert furthest_point_feasible(em, spec, d_exact * 1.001, ALT) is False  # just past -> infeasible


def test_min_work_allowance_can_flip_feasibility():
    spec = _multirotor_spec(360_000.0)
    em = EnergyModel(spec)
    d = 1_000.0
    assert furthest_point_feasible(em, spec, d, ALT, min_work_j=0.0) is True
    # demand more coverage work than the remaining headroom -> infeasible
    assert furthest_point_feasible(em, spec, d, ALT, min_work_j=1.0e9) is False


def test_tiny_battery_is_infeasible_even_nearby():
    spec = _multirotor_spec(1_000.0)  # absurdly small pack
    em = EnergyModel(spec)
    assert furthest_point_feasible(em, spec, 50.0, ALT) is False


# --------------------------------------------------------------------------- #
# pure: furthest free-space vertex distance                                   #
# --------------------------------------------------------------------------- #
def test_furthest_free_vertex_dist_on_a_square():
    area = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    env = EnvironmentMap(area, [], buffer_m=5.0)
    # from a corner, the far corner is the diagonal ~ 1414 m
    assert _furthest_free_vertex_dist(env, (0.0, 0.0)) == pytest.approx(math.hypot(1000, 1000), rel=1e-6)
    # from the centre, the furthest vertex is half the diagonal ~ 707 m
    assert _furthest_free_vertex_dist(env, (500.0, 500.0)) == pytest.approx(math.hypot(500, 500), rel=1e-6)


# --------------------------------------------------------------------------- #
# integration: optimize() success + InfeasibleMissionError                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def square_scenario():
    """Synthetic 1 km square (no obstacles) wired through the real GVG/TGC and
    the default platform/aero/launch config. Replace `load_config(CONFIG_PATH)`
    with your project's config fixture if the path differs."""
    cfg = load_config(CONFIG_PATH)
    spec = build_spec(cfg)
    em = EnergyModel(spec)
    motion = make_motion_model(spec)
    aero = AeroCorrection(cfg.aero, spec.platform)
    area = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    env = EnvironmentMap(area, [], cfg.env.clearance_buffer_m)
    gvg = build_gvg(env, sample_step_m=20.0, spur_min_m=30.0)
    tgc = build_tgc(env, gvg)
    rng = np.random.default_rng(0)
    return cfg, spec, em, motion, aero, env, tgc, rng


def test_optimize_returns_a_feasible_site(square_scenario):
    cfg, spec, em, motion, aero, env, tgc, rng = square_scenario
    pose, scores = optimize(cfg.launch, tgc, env, motion, em, aero, spec, 4, rng, ALT)
    assert isinstance(pose, Pose)
    assert scores  # candidates were scored
    # the chosen site must itself pass the feasibility gate
    furthest = _furthest_free_vertex_dist(env, pose.as_xy())
    assert furthest_point_feasible(em, spec, furthest, ALT) is True


def test_optimize_raises_when_no_site_is_reachable(square_scenario):
    cfg, spec, em, motion, aero, env, tgc, rng = square_scenario
    # shrink the battery so even the nearest site cannot reach the far bounds
    tiny_spec = dataclasses.replace(spec, battery_capacity_j=1_000.0)
    with pytest.raises(InfeasibleMissionError):
        optimize(cfg.launch, tgc, env, motion, em, aero, tiny_spec, 4, rng, ALT)
