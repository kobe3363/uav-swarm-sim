"""End-to-end smoke test (blueprint tests/test_smoke.py).

A tiny mission must run, complete, cover its area, end all agents in S0, and
produce a valid stationary distribution. Also checks the determinism contract.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import config_path

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState, DecompositionAlgo, PlannerKind
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.metrics.smdp_estimator import estimate
from uav_swarm_sim.metrics.stationary_distribution import stationary


def _tiny_cfg():
    # small area + ample battery + few obstacles: completes on one charge, fast,
    # deterministic, no swaps/failures on the critical path.
    return load_config(
        config_path(),
        overrides={
            "fleet.n_drones": 3,
            "fleet.battery_capacity_wh": 400.0,
            "failure.hazard_rate_per_hour": 0.0,
            "env.geojson_path": "data/areas/smoke_area.geojson",
            "env.obstacle_density_per_km2": 4.0,
            "env.obstacle_size_range_m": [10.0, 30.0],
            "sim.dt_s": 1.0,
            "sim.max_timesteps": 20000,
        },
    )


def test_mission_completes_and_covers():
    cfg = _tiny_cfg()
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()
    assert not result.aborted
    assert result.coverage_frac > 0.99
    assert result.metrics.total_energy_j > 0
    # every agent ends idle at base
    for a in eng.fleet.agents.values():
        assert a.state is AgentState.S0_IDLE


def test_stationary_distribution_valid():
    cfg = _tiny_cfg()
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=1,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI)
    result = eng.run()
    est = estimate(result.history, close_failure_loop=True)
    assert est.ergodic
    pi_emb, pi_time = stationary(est)
    assert pi_time.sum() == pytest.approx(1.0, abs=1e-9)
    assert (pi_time >= -1e-12).all()


def test_determinism_same_seed_same_result():
    cfg = _tiny_cfg()
    r1 = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                          algo=DecompositionAlgo.WEIGHTED_VORONOI).run()
    r2 = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                          algo=DecompositionAlgo.WEIGHTED_VORONOI).run()
    assert r1.config_hash == r2.config_hash
    assert r1.metrics.total_energy_j == pytest.approx(r2.metrics.total_energy_j)
    assert r1.metrics.duration_s == pytest.approx(r2.metrics.duration_s)


def test_grid_planner_mission_runs():
    cfg = _tiny_cfg()
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 0,
                           algo=DecompositionAlgo.WEIGHTED_VORONOI, planner=PlannerKind.GRID)
    result = eng.run()
    assert result.metrics.total_energy_j > 0
