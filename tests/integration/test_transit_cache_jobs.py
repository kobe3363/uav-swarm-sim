"""E3 x E2: the always-on visibility-graph cache must not break the --jobs
determinism gate. Each spawn worker builds its own per-replication cache
(instance attribute on a fresh engine, no shared memory), so serial and parallel
demand batches must stay byte-identical with the cache active.

This extends the E2 gate (tests/integration/test_spare_demand_jobs.py) with
``transit_free_space=True`` and a denser obstacle field, so route_transit -- and
thus the cache -- is on the execution path under both worker counts.
"""
from __future__ import annotations

import dataclasses

import pytest
from shapely.geometry import box

from uav_swarm_sim.experiments._parallel import run_units
from uav_swarm_sim.experiments.run_spare_sizing import run_demand
from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.core_types import Pose
from uav_swarm_sim.infrastructure.enums import ManeuverType
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.physical_model.drone_specs import build_spec
from uav_swarm_sim.physical_model.motion_model import make_motion_model
from uav_swarm_sim.planning.environment_map import EnvironmentMap
from uav_swarm_sim.planning.obstacle_generator import Obstacle
from uav_swarm_sim.planning.visibility_router import route_transit


def _tiny_routed_cfg(config_path):
    cfg = load_config(str(config_path), overrides={
        "fleet.n_drones": 2,
        "fleet.battery_capacity_wh": 20.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 8.0,
        "env.obstacle_size_range_m": [20.0, 60.0],
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
        "telemetry.enabled": False,
    })
    return dataclasses.replace(cfg, coverage=dataclasses.replace(
        cfg.coverage, transit_free_space=True))


def _by_rep(records):
    return sorted(records, key=lambda r: r.replication)


def _cache_exercised_in_spawn_worker(config_path: str):
    """Picklable, deterministic blocked-chord probe for ``run_units`` workers."""
    motion = make_motion_model(build_spec(load_config(config_path)))
    env = EnvironmentMap(
        box(0.0, 0.0, 1000.0, 1000.0),
        [Obstacle(id=0, cls=0, polygon=box(450.0, 400.0, 550.0, 600.0))],
        5.0,
    )
    a = Pose(200.0, 500.0, 0.0)
    b = Pose(800.0, 500.0, 0.0)
    cache: dict = {}
    routed = route_transit(a, b, motion, env, enabled=True, graph_cache=cache)
    chord = motion.plan(a, b, ManeuverType.CRUISE)
    return bool(cache), routed.total_length_m > chord.total_length_m


@pytest.mark.slow
def test_routed_demand_jobs_serial_parallel_byte_identical(config_path):
    """With the cache active (transit_free_space ON), --jobs 1 (serial) vs
    --jobs 2 (spawn) produce byte-identical DemandRecords. The per-worker cache
    (WKB-hash keyed, sha1 not salted) is process-stable, so completion order and
    worker count cannot perturb the result."""
    cfg = _tiny_routed_cfg(config_path)
    reps = 4
    serial = run_demand(cfg, reps, RngFactory(cfg.sim.master_seed), jobs=1)
    parallel = run_demand(cfg, reps, RngFactory(cfg.sim.master_seed), jobs=2)
    assert _by_rep(serial) == _by_rep(parallel)

    # ``transit_free_space=True`` installs the planner but does not guarantee a
    # random demand replication has a blocked chord.  A deterministic probe uses
    # the same spawn helper and requires an actual cached obstacle detour.
    probe_args = [(str(config_path),), (str(config_path),)]
    expected = [(True, True), (True, True)]
    assert run_units(_cache_exercised_in_spawn_worker, probe_args, jobs=1) == expected
    assert sorted(run_units(_cache_exercised_in_spawn_worker, probe_args, jobs=2)) == expected
