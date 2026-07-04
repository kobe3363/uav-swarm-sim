"""Tests for the target-visit mission type (allocation, routing, end-to-end)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState, DecompositionAlgo, MissionType
from uav_swarm_sim.infrastructure.core_types import DroneStateView, Pose
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.metrics.smdp_estimator import estimate
from uav_swarm_sim.metrics.stationary_distribution import stationary
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.energy_model import EnergyModel
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.planning import target_mission as tm
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.geojson_parser import load_area
from uav_swarm_sim.planning.obstacle_generator import generate as gen_obstacles


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def env(config_path):
    cfg = load_config(config_path, overrides={"env.geojson_path": "data/areas/smoke_area.geojson",
                                                 "env.obstacle_density_per_km2": 4.0})
    area = load_area(cfg.env.geojson_path)
    rng = RngFactory(cfg.sim.master_seed).stream("obstacles", 0)
    obstacles = gen_obstacles(area, cfg.env, rng)
    return EnvironmentMap(area, obstacles, cfg.env.clearance_buffer_m)


@pytest.fixture(scope="module")
def specs(config_path):
    cfg = load_config(config_path)
    spec = build_spec(cfg)
    return spec, make_motion_model(spec), EnergyModel(spec)


def _views(n, batteries=None):
    return [DroneStateView(i, 1.0 if batteries is None else batteries[i], Pose(0, 0, 0)) for i in range(n)]


# --------------------------------------------------------------------------- #
# generation                                                                  #
# --------------------------------------------------------------------------- #
def test_random_targets_count_and_in_free_space(config_path, env):
    cfg = load_config(config_path, overrides={"mission.type": "target_visit", "mission.n_targets": 20})
    rng = RngFactory(1).stream("targets", 0)
    pts = tm.generate_targets(env, cfg.mission, rng)
    assert len(pts) == 20
    for p in pts:
        assert env.contains(p)


def test_random_targets_reproducible(config_path, env):
    cfg = load_config(config_path, overrides={"mission.type": "target_visit", "mission.n_targets": 15})
    a = tm.generate_targets(env, cfg.mission, RngFactory(7).stream("targets", 3))
    b = tm.generate_targets(env, cfg.mission, RngFactory(7).stream("targets", 3))
    assert a == b


def test_explicit_targets_used_verbatim(config_path, env):
    coords = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    cfg = load_config(config_path, overrides={"mission.type": "target_visit",
                                                "mission.target_coordinates": coords})
    pts = tm.generate_targets(env, cfg.mission, RngFactory(0).stream("targets", 0))
    assert pts == [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]


# --------------------------------------------------------------------------- #
# allocation (battery-weighted)                                               #
# --------------------------------------------------------------------------- #
def test_capacities_sum_to_total_and_even_when_equal():
    caps = tm.capacities(_views(4), 20, weight_by_battery=True)
    assert sum(caps) == 20
    assert max(caps) - min(caps) <= 1  # equal batteries -> as even as possible


def test_capacities_weighted_by_battery():
    caps = tm.capacities(_views(3, batteries=[1.0, 0.5, 0.25]), 28, weight_by_battery=True)
    assert sum(caps) == 28
    assert caps[0] > caps[1] > caps[2]  # more battery -> more targets


def test_assign_covers_every_target_once():
    targets = [(float(i), float(i % 5)) for i in range(17)]
    views = _views(3)
    assignment = tm.assign_targets(targets, views, launch=(0.0, 0.0), weight_by_battery=True)
    flat = [p for pts in assignment.values() for p in pts]
    assert len(flat) == len(targets)
    assert set(flat) == set(targets)             # exact partition, no loss/dup
    counts = [len(assignment[v.id]) for v in views]
    assert counts == tm.capacities(views, 17, True)


# --------------------------------------------------------------------------- #
# routing                                                                      #
# --------------------------------------------------------------------------- #
def test_route_tour_visits_all_targets(specs):
    spec, motion, em = specs
    pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (50.0, 50.0)]
    plan = tm.route_tour(0, entry=(0.0, 0.0), pts=pts, spec=spec, em=em)
    assert plan.leg_mode == "tour"
    assert len(plan.waypoints) == len(pts)
    visited = {(round(w.pose.x, 6), round(w.pose.y, 6)) for w in plan.waypoints}
    assert visited == {(round(x, 6), round(y, 6)) for x, y in pts}
    assert plan.length_m > 0 and plan.est_energy_j > 0


def test_route_tour_empty_is_safe(specs):
    spec, motion, em = specs
    plan = tm.route_tour(0, entry=(0.0, 0.0), pts=[], spec=spec, em=em)
    assert plan.waypoints == [] and plan.length_m == 0.0


def test_plan_target_mission_partition_and_plans(env, specs):
    spec, motion, em = specs
    targets = env.sample_free(18, RngFactory(2).stream("targets", 0))
    views = _views(3)
    part, plans, assignment = tm.plan_target_mission(targets, views, Pose(0, 0, 0), motion, spec, em)
    assert set(plans) == {0, 1, 2}
    assert sum(len(v) for v in assignment.values()) == len(targets)
    assert part.total_area_m2 > 0   # hull zones exist for visualization


# --------------------------------------------------------------------------- #
# end-to-end engine                                                           #
# --------------------------------------------------------------------------- #
def _target_cfg(config_path, **extra):
    ov = {
        "mission.type": "target_visit",
        "mission.n_targets": 24,
        "fleet.n_drones": 3,
        "fleet.battery_capacity_wh": 400.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "env.obstacle_size_range_m": [10.0, 30.0],
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
    }
    ov.update(extra)
    return load_config(config_path, overrides=ov)


def test_target_mission_completes_and_visits_all(config_path):
    cfg = _target_cfg(config_path)
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()
    assert eng._mission_type is MissionType.TARGET_VISIT
    assert len(eng.targets) == 24
    assert sum(len(v) for v in eng.assignment.values()) == 24
    assert not result.aborted
    assert result.coverage_frac > 0.99          # all targets visited
    assert result.metrics.total_energy_j > 0
    for a in eng.fleet.agents.values():
        assert a.state is AgentState.S0_IDLE


def test_target_mission_smdp_valid(config_path):
    cfg = _target_cfg(config_path)
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 1,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()
    est = estimate(result.history, close_failure_loop=True)
    assert est.ergodic
    _, pi_time = stationary(est)
    assert abs(pi_time.sum() - 1.0) < 1e-9       # same analysis stack, unchanged


def test_target_mission_deterministic(config_path):
    cfg = _target_cfg(config_path)
    r1 = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                          algo=DecompositionAlgo.WEIGHTED_VORONOI).run()
    r2 = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                          algo=DecompositionAlgo.WEIGHTED_VORONOI).run()
    assert r1.metrics.total_energy_j == pytest.approx(r2.metrics.total_energy_j)
    assert r1.metrics.duration_s == pytest.approx(r2.metrics.duration_s)


def test_explicit_coordinates_mission_runs(config_path):
    coords = [[80, 80], [320, 80], [320, 220], [80, 220], [200, 150], [150, 100]]
    cfg = _target_cfg(config_path, **{"mission.target_coordinates": coords})
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()
    assert len(eng.targets) == 6
    assert not result.aborted and result.coverage_frac > 0.99


# --------------------------------------------------------------------------- #
# config switch                                                               #
# --------------------------------------------------------------------------- #
def test_default_mission_is_coverage(config_path):
    cfg = load_config(config_path)
    assert cfg.mission.type is MissionType.COVERAGE


def test_coverage_mission_still_works(config_path):
    cfg = load_config(config_path, overrides={
        "fleet.n_drones": 3, "fleet.battery_capacity_wh": 400.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0, "sim.dt_s": 1.0, "sim.max_timesteps": 20000,
    })
    assert cfg.mission.type is MissionType.COVERAGE
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()
    assert not result.aborted and result.coverage_frac > 0.99
