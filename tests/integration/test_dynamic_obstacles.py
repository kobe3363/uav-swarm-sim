"""Tests for dynamic obstacles + swarm passive/active sensing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo, SensingMode
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.infrastructure import visualization as viz
from uav_swarm_sim.planning.dynamic_obstacles import DynamicObstacleField
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.geojson_parser import load_area
from uav_swarm_sim.planning.obstacle_generator import generate as gen_obstacles
from uav_swarm_sim.execution.sensing import SensingCoordinator


@pytest.fixture(scope="module")
def env(config_path):
    cfg = load_config(config_path, overrides={"env.geojson_path": "data/areas/smoke_area.geojson",
                                                 "env.obstacle_density_per_km2": 3.0})
    area = load_area(cfg.env.geojson_path)
    rng = RngFactory(cfg.sim.master_seed).stream("obstacles", 0)
    return EnvironmentMap(area, gen_obstacles(area, cfg.env, rng), cfg.env.clearance_buffer_m)


# --------------------------------------------------------------------------- #
# the moving-obstacle field                                                   #
# --------------------------------------------------------------------------- #
def test_field_spawns_in_free_space_with_speed(env):
    rng = RngFactory(1).stream("dynamic_obstacles", 0)
    field = DynamicObstacleField(env, count=5, speed_m_s=10.0, size_m=1.5, rng=rng)
    assert len(field.obstacles) == 5
    for o in field.obstacles:
        assert env.contains((o.x, o.y))                       # spawned in free space
        assert abs((o.vx ** 2 + o.vy ** 2) ** 0.5 - 10.0) < 1e-6  # configured speed
        assert o.radius == pytest.approx(0.75)                 # drone-sized (diameter 1.5)


def test_field_moves_and_stays_in_bounds(env):
    rng = RngFactory(2).stream("dynamic_obstacles", 0)
    field = DynamicObstacleField(env, count=4, speed_m_s=20.0, size_m=1.5, rng=rng)
    before = [(o.x, o.y) for o in field.obstacles]
    minx, miny, maxx, maxy = env.area.bounds
    for _ in range(500):
        field.step(1.0)
    after = [(o.x, o.y) for o in field.obstacles]
    assert before != after                                    # they actually move
    for o in field.obstacles:                                 # reflection keeps them in the box
        assert minx - 1 <= o.x <= maxx + 1 and miny - 1 <= o.y <= maxy + 1


def test_field_reproducible(env):
    a = DynamicObstacleField(env, 3, 12.0, 1.5, RngFactory(9).stream("dynamic_obstacles", 1))
    b = DynamicObstacleField(env, 3, 12.0, 1.5, RngFactory(9).stream("dynamic_obstacles", 1))
    assert [(o.x, o.y, o.vx, o.vy) for o in a.obstacles] == [(o.x, o.y, o.vx, o.vy) for o in b.obstacles]


# --------------------------------------------------------------------------- #
# swarm passive/active sensing logic (coordinator in isolation)               #
# --------------------------------------------------------------------------- #
class _StubAgent:
    def __init__(self, i, x, y):
        self.id = i
        from uav_swarm_sim.infrastructure.core_types import Pose
        from uav_swarm_sim.infrastructure.enums import AgentState
        self.pose = Pose(x, y, 0.0)
        self.state = AgentState.S2_MISSION
        self.threatened = False
    def signal_threat(self, on):
        self.threatened = self.threatened or on


class _StubField:
    def __init__(self, pts):
        from uav_swarm_sim.planning.dynamic_obstacles import DynamicObstacle
        self.obstacles = [DynamicObstacle(i, x, y, 0, 0, 0.75) for i, (x, y) in enumerate(pts)]


class _Bus:
    def __init__(self): self.events = []
    def publish(self, e): self.events.append(e)


def _cfg(config_path, **ov):
    base = {
        "dynamic_obstacles.enabled": True, "dynamic_obstacles.count": 1,
        "dynamic_obstacles.passive_sense_range_m": 30.0,
        "dynamic_obstacles.active_sense_range_m": 200.0,
        "dynamic_obstacles.active_scan_power_w": 80.0,
        "dynamic_obstacles.dynamic_hold_s": 10.0,
    }
    base.update(ov)
    return load_config(config_path, overrides=base)


def test_passive_until_detected_then_whole_swarm_active(config_path):
    cfg = _cfg(config_path)
    co = SensingCoordinator(cfg.dynamic_obstacles, cfg.safety)
    assert co.mode is SensingMode.PASSIVE and co.scan_power_w() == 0.0
    agents = [_StubAgent(0, 0, 0), _StubAgent(1, 1000, 1000)]   # agent 1 far away
    bus = _Bus()
    # obstacle 25 m from agent 0 (< passive range 30) -> swarm flips ACTIVE
    co.step(agents, _StubField([(25.0, 0.0)]), t=1.0, bus=bus)
    assert co.mode is SensingMode.ACTIVE
    assert co.scan_power_w() == pytest.approx(80.0)            # scanning now costs power


def test_undetected_obstacle_is_ignored(config_path):
    cfg = _cfg(config_path)
    co = SensingCoordinator(cfg.dynamic_obstacles, cfg.safety)
    agents = [_StubAgent(0, 0, 0)]
    bus = _Bus()
    # obstacle 100 m away, beyond passive range (30) -> nobody notices, stays passive
    co.step(agents, _StubField([(100.0, 0.0)]), t=1.0, bus=bus)
    assert co.mode is SensingMode.PASSIVE
    assert not agents[0].threatened


def test_reverts_to_passive_after_hold(config_path):
    cfg = _cfg(config_path, **{"dynamic_obstacles.dynamic_hold_s": 5.0})
    co = SensingCoordinator(cfg.dynamic_obstacles, cfg.safety)
    agents = [_StubAgent(0, 0, 0)]
    bus = _Bus()
    co.step(agents, _StubField([(20.0, 0.0)]), t=0.0, bus=bus)
    assert co.mode is SensingMode.ACTIVE
    # obstacle now far beyond even active range; after hold elapses -> passive
    co.step(agents, _StubField([(10000.0, 0.0)]), t=10.0, bus=bus)
    assert co.mode is SensingMode.PASSIVE


def test_near_miss_signals_threat(config_path):
    cfg = _cfg(config_path)
    co = SensingCoordinator(cfg.dynamic_obstacles, cfg.safety)
    a = _StubAgent(0, 0, 0)
    bus = _Bus()
    # within min_separation -> agent gets an avoidance threat + an event is published
    co.step([a], _StubField([(2.0, 0.0)]), t=1.0, bus=bus)
    assert a.threatened
    assert any("dynamic_obstacle_id" in e.payload for e in bus.events)


# --------------------------------------------------------------------------- #
# central on/off + end-to-end energy cost                                     #
# --------------------------------------------------------------------------- #
def _engine_cfg(config_path, enabled):
    return load_config(config_path, overrides={
        "dynamic_obstacles.enabled": enabled, "dynamic_obstacles.count": 4,
        "dynamic_obstacles.speed_m_s": 10.0, "dynamic_obstacles.active_scan_power_w": 120.0,
        "dynamic_obstacles.passive_sense_range_m": 40.0, "dynamic_obstacles.active_sense_range_m": 200.0,
        "fleet.n_drones": 3, "fleet.battery_capacity_wh": 400.0,
        "failure.hazard_rate_per_hour": 0.0, "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 3.0, "sim.dt_s": 1.0, "sim.max_timesteps": 20000,
    })


def test_default_feature_is_off(config_path):
    assert load_config(config_path).dynamic_obstacles.enabled is False


def test_disabled_matches_baseline_and_no_field(config_path):
    cfg = _engine_cfg(config_path, False)
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0, algo=DecompositionAlgo.WEIGHTED_VORONOI)
    r = eng.run()
    assert eng._dynfield is None
    assert eng.sensing.mode is SensingMode.PASSIVE
    assert eng.history.dynamic_obstacle_frames() == []
    assert not r.aborted


def test_enabled_scans_costs_energy_and_records_frames(config_path):
    off = SimulationEngine(_engine_cfg(config_path, False), RngFactory(42), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI).run()
    eng_on = SimulationEngine(_engine_cfg(config_path, True), RngFactory(42), 0,
                              algo=DecompositionAlgo.WEIGHTED_VORONOI)
    on = eng_on.run()
    assert eng_on._dynfield is not None
    frames = eng_on.history.dynamic_obstacle_frames()
    assert len(frames) > 0
    assert any(mode == "ACTIVE" for (_t, mode, _obs) in frames)     # swarm got provoked
    assert on.metrics.total_energy_j > off.metrics.total_energy_j   # LIDAR scanning costs energy
    assert not on.aborted


def test_enabled_is_deterministic(config_path):
    r1 = SimulationEngine(_engine_cfg(config_path, True), RngFactory(42), 0, algo=DecompositionAlgo.WEIGHTED_VORONOI).run()
    r2 = SimulationEngine(_engine_cfg(config_path, True), RngFactory(42), 0, algo=DecompositionAlgo.WEIGHTED_VORONOI).run()
    assert r1.metrics.total_energy_j == pytest.approx(r2.metrics.total_energy_j)


@pytest.mark.slow
def test_replay_renders_dynamic_obstacles(config_path, tmp_path):
    eng = SimulationEngine(_engine_cfg(config_path, True), RngFactory(42), 0, algo=DecompositionAlgo.WEIGHTED_VORONOI)
    eng.run()
    gif = viz.animate_mission(eng.history, eng.env, tmp_path / "replay.gif", fps=8, max_frames=40)
    png = viz.plot_state_colored_paths(eng.history, eng.env, tmp_path / "paths.png")
    assert Path(gif).stat().st_size > 0 and Path(gif).read_bytes()[:6] in (b"GIF87a", b"GIF89a")
    assert Path(png).stat().st_size > 0
