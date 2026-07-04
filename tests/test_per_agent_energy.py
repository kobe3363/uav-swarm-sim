"""per_agent_energy_j on MissionMetrics (S5 prerequisite).

The S5 shape sweep's PRIMARY balance metric is the executed ENERGY imbalance
max(per-drone E)/mean(per-drone E). That needs the per-drone consumed energy,
which ``mission_metrics.compute`` previously reduced to ``total_energy_j``
only. The new ``per_agent_energy_j`` dict is appended with a default (so every
existing constructor call stays valid) and must:

  * carry one entry per fleet drone (keys == per_agent_length_m keys),
  * sum to ``total_energy_j`` exactly (single source: the same agent reads),
  * be non-negative per drone on a real mission.
"""
from __future__ import annotations

import dataclasses
import json
import math

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo, PlannerKind
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine


def _tiny_square_geojson(tmp_path, side_m: float = 220.0) -> str:
    """A small synthetic survey square so the mission finishes fast."""
    coords = [[0, 0], [side_m, 0], [side_m, side_m], [0, side_m], [0, 0]]
    gj = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }]}
    p = tmp_path / "tiny_square.geojson"
    p.write_text(json.dumps(gj), encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def tiny_mission_metrics(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("per_agent_energy")
    cfg = load_config("config/default.yaml")
    env = dataclasses.replace(cfg.env, geojson_path=_tiny_square_geojson(tmp),
                              obstacle_density_per_km2=0.0)
    fleet = dataclasses.replace(cfg.fleet, n_drones=2)
    failure = dataclasses.replace(cfg.failure, hazard_rate_per_hour=0.0)
    cfg = dataclasses.replace(cfg, env=env, fleet=fleet, failure=failure)
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), 1,
                           algo=DecompositionAlgo.TGC_BASIC,
                           planner=PlannerKind.DUBINS)
    result = eng.run()
    return result.metrics


def test_per_agent_energy_keys_match_fleet(tiny_mission_metrics):
    m = tiny_mission_metrics
    assert set(m.per_agent_energy_j.keys()) == set(m.per_agent_length_m.keys())
    assert len(m.per_agent_energy_j) == 2


def test_per_agent_energy_sums_to_total(tiny_mission_metrics):
    m = tiny_mission_metrics
    assert math.isclose(sum(m.per_agent_energy_j.values()), m.total_energy_j,
                        rel_tol=0.0, abs_tol=1e-9)


def test_per_agent_energy_nonnegative_and_spent(tiny_mission_metrics):
    m = tiny_mission_metrics
    assert all(e >= 0.0 for e in m.per_agent_energy_j.values())
    assert m.total_energy_j > 0.0  # a real mission burns energy


def test_default_keeps_existing_constructors_valid():
    """Appended-with-default: constructing MissionMetrics WITHOUT the new field
    (every pre-existing call site) still works and yields an empty dict."""
    from uav_swarm_sim.metrics.mission_metrics import MissionMetrics
    m = MissionMetrics(total_energy_j=1.0, duration_s=2.0, workload_std_m=0.0,
                       per_agent_length_m={0: 1.0}, n_swaps=0, n_failures=0,
                       coverage_frac=1.0, planning_time_s=0.0)
    assert m.per_agent_energy_j == {}
