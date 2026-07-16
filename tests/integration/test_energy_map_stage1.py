"""EM-01 Stage 1 gate: the energy map is built-and-attached ONLY -- toggling it
changes NO mission metric (the acceptance gate that matters most).

Flag OFF (the default; ``rth.energy_map`` is absent from default.yaml) leaves
the engine's build path structurally untouched -- a single ``self.energy_map =
None`` assignment, no RNG draw, no physics -- so flag-off runs are byte-identical
to pre-change main by construction. The executable check below follows the
repo's established gate pattern (test_sweep_telemetry_off.py:
``test_telemetry_on_off_metrics_byte_identical``): the SAME tiny mission with
the flag ON vs OFF must produce the identical metrics tuple, since Stage 1
wires no consumer. It also pins the Stage-1 contract itself: OFF -> no map;
ON -> a built ``EnergyMap`` with ``E_home[base] == 0``.
"""
from __future__ import annotations

import dataclasses

import pytest

from uav_swarm_sim.infrastructure.config import EnergyMapConfig, load_config
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo, PlannerKind
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine
from uav_swarm_sim.planning.energy_map import EnergyMap


def _tiny_cfg(config_path):
    return load_config(str(config_path), overrides={
        "fleet.n_drones": 2,
        "fleet.battery_capacity_wh": 400.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
        "telemetry.enabled": False,
    })


def _run(cfg):
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=0,
                           algo=DecompositionAlgo.TGC_BASIC,
                           planner=PlannerKind.DUBINS)
    m = eng.run().metrics
    return eng, (m.total_energy_j, m.duration_s, m.workload_std_m, m.n_swaps,
                 dict(m.per_agent_energy_j), dict(m.per_agent_length_m))


@pytest.mark.slow
def test_energy_map_on_off_metrics_byte_identical(config_path):
    base = _tiny_cfg(config_path)
    assert base.rth.energy_map.enabled is False   # guards the shipped default
    cfg_on = dataclasses.replace(
        base, rth=dataclasses.replace(base.rth, energy_map=EnergyMapConfig(enabled=True)))

    eng_off, metrics_off = _run(base)
    eng_on, metrics_on = _run(cfg_on)

    assert metrics_on == metrics_off              # Stage 1 consumes nothing

    # Stage-1 contract: OFF -> nothing built; ON -> map built and attached.
    assert eng_off.energy_map is None
    emap = eng_on.energy_map
    assert isinstance(emap, EnergyMap)
    bi, bj = emap.frame.world_to_cell(eng_on.launch_pose.x, eng_on.launch_pose.y)
    assert emap.e_home[bi, bj] == 0.0
    assert emap.parent[bi, bj] == -1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
