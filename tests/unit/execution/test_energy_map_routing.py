"""EM-01 Stage 3 -- map-routed S3 return + resume transit (rth.energy_map.route).

Unit tests for the two Stage-3 seams on tiny synthetic environments (env
scaffolding mirrors test_rth_map_decide.py; wall geometry mirrors
test_transit_routing.py):

  * 7c plan_return: a wall between the drone and the base -> the returned Path
    goes AROUND (no sampled span crosses a raw obstacle, length > the straight
    chord) and ends at the base;
  * 7d plan_resume: the reversed parent chain base -> entry, same clearance
    guarantee, arrival keeps the entry heading (matching the straight chord's
    arrival exactly);
  * fallback contract: an E_home=inf cell (sealed pocket) or an out-of-grid
    pose -> None + n_route_fallbacks, never a crash;
  * precedence (seam 7d vs FIX-B1): map ON -> plan_resume wins and the injected
    _transit_planner is NOT consulted; map OFF -> the B1 planner is consulted
    exactly as before (byte-identical structure);
  * config schema: ``route`` defaults False; ``route`` without ``enabled`` is a
    ConfigError (same rule as ``decide``).
"""
from __future__ import annotations

import dataclasses
import math

import pytest
from shapely.geometry import Polygon, box

from uav_swarm_sim.execution.agent import Agent
from uav_swarm_sim.execution.rth_calculator import RthCalculator
from uav_swarm_sim.execution.state_machine import StateMachine
from uav_swarm_sim.infrastructure.config import ConfigError, EnergyMapConfig, load_config
from uav_swarm_sim.infrastructure.core_types import CoveragePlan, Pose, Waypoint
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.physical_model.aero_correction import AeroCorrection
from uav_swarm_sim.physical_model.battery import Battery
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.execution.formation_manager import FormationManager
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.planning.energy_map import build_energy_map
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.obstacle_generator import Obstacle

AREA = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
CELL = 20.0
ALT = 100.0
BASE = Pose(100.0, 500.0, 0.0)


@pytest.fixture(scope="module")
def kit(config_path):
    cfg = load_config(config_path, overrides={"platform_type": "MULTIROTOR"})
    spec = build_spec(cfg)
    return cfg, spec, make_motion_model(spec), EnergyModel(spec)


def _env(obstacles):
    return EnvironmentMap(AREA, obstacles, 5.0)


def _calc(kit, env, base=BASE, route=True):
    cfg, spec, motion, em = kit
    emap = build_energy_map(env, base, CELL, em, spec.v_cruise)
    rth_cfg = dataclasses.replace(
        cfg.rth, energy_map=EnergyMapConfig(enabled=True, route=route))
    return RthCalculator(em, motion, spec, rth_cfg, base, altitude_m=ALT,
                         env=env, energy_map=emap)


def _spans_clear(path, env, ds=5.0) -> bool:
    pts = path.sample(ds)
    return all(not env.segment_in_obstacle(pts[i], pts[i + 1])
               for i in range(len(pts) - 1))


# --------------------------------------------------------------------------- #
# seam 7c: return routing around a wall                                        #
# --------------------------------------------------------------------------- #
def test_plan_return_routes_around_wall(kit):
    _, _, motion, _ = kit
    env = _env([Obstacle(id=0, cls=0, polygon=box(480, 300, 520, 700))])
    calc = _calc(kit, env)
    pose = Pose(900.0, 500.0, math.pi)
    assert env.segment_in_obstacle(pose, BASE)  # premise: the chord is blocked

    path = calc.plan_return(pose)
    assert path is not None
    assert calc.n_route_fallbacks == 0
    chord = motion.plan(pose, BASE, ManeuverType.CRUISE)
    assert path.total_length_m > chord.total_length_m
    assert _spans_clear(path, env)
    end = path.end_pose
    assert math.dist(end.as_xy(), BASE.as_xy()) < 1e-6
    assert end.heading == BASE.heading


def test_plan_return_free_space_stays_near_geodesic(kit):
    # no obstacles: the routed polyline is the grid geodesic -- never shorter
    # than the chord, and within the octile-anisotropy envelope of it
    _, _, motion, _ = kit
    env = _env([])
    calc = _calc(kit, env)
    pose = Pose(900.0, 500.0, math.pi)
    path = calc.plan_return(pose)
    assert path is not None
    chord = motion.plan(pose, BASE, ManeuverType.CRUISE)
    assert path.total_length_m >= chord.total_length_m - 1e-9
    assert path.total_length_m <= chord.total_length_m * 1.0824 + 2 * CELL


# --------------------------------------------------------------------------- #
# seam 7d: resume routing (reversed chain)                                     #
# --------------------------------------------------------------------------- #
def test_plan_resume_routes_around_wall_and_keeps_entry_heading(kit):
    env = _env([Obstacle(id=0, cls=0, polygon=box(480, 300, 520, 700))])
    calc = _calc(kit, env)
    entry = Pose(900.0, 500.0, 1.25)
    assert env.segment_in_obstacle(BASE, entry)

    path = calc.plan_resume(BASE, entry)
    assert path is not None
    assert _spans_clear(path, env)
    assert math.dist(path.start_pose.as_xy(), BASE.as_xy()) < 1e-6
    end = path.end_pose
    assert math.dist(end.as_xy(), entry.as_xy()) < 1e-6
    assert end.heading == entry.heading  # same arrival heading as the chord


# --------------------------------------------------------------------------- #
# fallback contract                                                            #
# --------------------------------------------------------------------------- #
def test_sealed_pocket_and_out_of_grid_fall_back_counted(kit):
    # a closed ring of 20 m walls (buffered to ~30 m -> red cells all around)
    # seals the pocket at (800, 800): E_home = inf inside
    ring = [
        Obstacle(id=0, cls=0, polygon=box(700, 700, 900, 720)),
        Obstacle(id=1, cls=0, polygon=box(700, 880, 900, 900)),
        Obstacle(id=2, cls=0, polygon=box(700, 700, 720, 900)),
        Obstacle(id=3, cls=0, polygon=box(880, 700, 900, 900)),
    ]
    env = _env(ring)
    calc = _calc(kit, env)
    boxed = Pose(800.0, 800.0, 0.0)
    i, j = calc._map.frame.world_to_cell(boxed.x, boxed.y)
    assert not math.isfinite(calc._map.e_home[i, j])  # premise: truly sealed

    assert calc.plan_return(boxed) is None
    assert calc.n_route_fallbacks == 1
    assert calc.plan_resume(BASE, boxed) is None
    assert calc.n_route_fallbacks == 2

    outside = Pose(5000.0, 5000.0, 0.0)
    assert calc.plan_return(outside) is None
    assert calc.n_route_fallbacks == 3


# --------------------------------------------------------------------------- #
# precedence: map vs FIX-B1 on the resume path                                 #
# --------------------------------------------------------------------------- #
def _agent_with_planner(kit, cfg_path, route: bool, transit_planner):
    cfg, spec, motion, em = kit
    env = _env([])
    if route:
        rth = _calc(kit, env, base=Pose(0.0, 0.0, 0.0), route=True)
    else:
        rth = RthCalculator(em, motion, spec, cfg.rth, Pose(0.0, 0.0, 0.0),
                            altitude_m=ALT)
    bat = Battery(spec.battery_capacity_j, cfg.battery_zones, 1.0)
    sm = StateMachine(cfg.battery_zones)
    aero = AeroCorrection(cfg.aero, spec.platform)
    fm = FormationManager(aero, cfg.aero, spec.platform)
    base = Pose(0.0, 0.0, 0.0)
    agent = Agent(0, spec, motion, em, bat, sm, rth, fm, base,
                  transit_planner=transit_planner)
    wps = [
        Waypoint(Pose(100, 0, 0.0), ManeuverType.COVERAGE, 6.0),
        Waypoint(Pose(150, 0, 0.0), ManeuverType.COVERAGE, 6.0),
    ]
    agent.assign(CoveragePlan(0, wps, 0.0, 0.0),
                 motion.plan(base, Pose(100, 0, 0.0), ManeuverType.CRUISE))
    return agent


def test_resume_precedence_map_beats_b1(kit, config_path):
    _, _, motion, _ = kit
    sentinel = motion.plan(Pose(0, 0, 0.0), Pose(1, 1, 0.0), ManeuverType.CRUISE)
    calls: list[tuple] = []

    def b1_spy(a, b):
        calls.append((a, b))
        return sentinel

    on = _agent_with_planner(kit, config_path, route=True, transit_planner=b1_spy)
    routed = on._resume_transit()
    assert routed is not sentinel      # the map produced the path
    assert calls == []                 # B1 was never consulted
    assert on.rth.n_route_fallbacks == 0

    off = _agent_with_planner(kit, config_path, route=False, transit_planner=b1_spy)
    resumed = off._resume_transit()
    assert resumed is sentinel         # flag off -> exactly the pre-Stage-3 B1 path
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# config schema                                                                #
# --------------------------------------------------------------------------- #
def test_route_defaults_false_and_requires_enabled(config_path):
    assert load_config(config_path).rth.energy_map.route is False
    with pytest.raises(ConfigError, match="route requires"):
        load_config(config_path, overrides={"rth.energy_map.route": True})
    cfg = load_config(config_path, overrides={
        "rth.energy_map.enabled": True, "rth.energy_map.route": True})
    assert cfg.rth.energy_map.route is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
