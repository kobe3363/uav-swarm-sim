"""EM-01 Stage 2 gate: the ``decide`` sub-flag owns ALL behaviour change.

``enabled=True, decide=False`` must stay byte-identical to ``enabled=False``
(the Stage-1 build-and-attach contract, now pinned against Stage 2's own
flag), while ``decide=True`` runs the map-based decide with the per-sortie
arming invariant holding end-to-end: the shared calculator's ``decide`` is
never invoked at a battery level above the caller's sortie arm.

Deliberately NOT asserted: that the ``rth_energy`` transition reason appears.
On a tiny 1 km2-class mission the design doc itself predicts the dynamic
trigger may never fire (that question belongs to the Stage-5 A/B study, not
to a gate test).
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from uav_swarm_sim.infrastructure.config import EnergyMapConfig, load_config
from uav_swarm_sim.infrastructure.enums import DecompositionAlgo, PlannerKind
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine


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


def _engine(cfg):
    return SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=0,
                            algo=DecompositionAlgo.TGC_BASIC,
                            planner=PlannerKind.DUBINS)


def _metrics_tuple(m):
    return (m.total_energy_j, m.duration_s, m.workload_std_m, m.n_swaps,
            dict(m.per_agent_energy_j), dict(m.per_agent_length_m))


@pytest.mark.slow
def test_decide_off_metrics_byte_identical(config_path):
    """enabled=True alone (decide default False) still consumes nothing."""
    base = _tiny_cfg(config_path)
    cfg_build = dataclasses.replace(
        base, rth=dataclasses.replace(base.rth, energy_map=EnergyMapConfig(enabled=True)))
    assert cfg_build.rth.energy_map.decide is False  # guards the shipped default

    m_off = _metrics_tuple(_engine(base).run().metrics)
    m_build = _metrics_tuple(_engine(cfg_build).run().metrics)
    assert m_build == m_off


@pytest.mark.slow
def test_decide_on_runs_and_arm_invariant_holds(config_path, monkeypatch):
    base = _tiny_cfg(config_path)
    cfg_on = dataclasses.replace(
        base, rth=dataclasses.replace(
            base.rth, energy_map=EnergyMapConfig(enabled=True, decide=True)))

    # class-level counting spy: the calculator is constructed inside
    # eng.run() -> _build(), so an instance wrapper cannot be installed ahead
    # of time; record (level, caller's arm) for every energy decide taken
    from uav_swarm_sim.execution.rth_calculator import RthCalculator

    seen: list[tuple[float, float]] = []
    orig_decide = RthCalculator.decide

    def spy(self, agent):
        seen.append((agent.battery.level_j, agent._arm_level_j))
        return orig_decide(self, agent)

    monkeypatch.setattr(RthCalculator, "decide", spy)

    eng = _engine(cfg_on)
    result = eng.run()
    assert eng.rth.map_decide_on
    assert result.outcome is not None  # ran to a terminal outcome

    # every launched agent armed its sortie(s); arms are finite and recorded
    launched = [a for a in eng.fleet.agents.values() if a.sortie_arms]
    assert launched, "no agent ever armed a sortie"
    for a in launched:
        for sortie_idx, arm_j in a.sortie_arms:
            assert sortie_idx >= 1
            assert math.isfinite(arm_j) and arm_j > 0.0

    # arming itself evaluates map-served return energies at the bundle ends
    assert eng.rth.n_map_hits > 0

    # the 7b invariant end-to-end: no decide ever ran above the sortie arm
    for level_j, arm_j in seen:
        assert level_j <= arm_j + 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
