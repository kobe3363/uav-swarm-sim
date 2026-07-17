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


def _tiny_cfg(config_path, battery_wh=400.0):
    return load_config(str(config_path), overrides={
        "fleet.n_drones": 2,
        "fleet.battery_capacity_wh": battery_wh,
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
    # small battery: the per-sortie arm sits ABOVE the static critical zone
    # (arm ~0.50 cap vs critical 0.20 at 15 Wh, probed empirically), so the
    # battery-quantized energy decide actually RUNS before the static net --
    # asserted non-vacuously below; with the 400 Wh smoke battery the level
    # never crosses the arm and the decide path would go untested
    base = _tiny_cfg(config_path, battery_wh=15.0)
    cfg_on = dataclasses.replace(
        base, rth=dataclasses.replace(
            base.rth, energy_map=EnergyMapConfig(enabled=True, decide=True)))

    # class-level counting spy: the calculator is constructed inside
    # eng.run() -> _build(), so an instance wrapper cannot be installed ahead
    # of time; record (level, caller's arm, per-call map-lookup delta)
    from uav_swarm_sim.execution.rth_calculator import RthCalculator

    seen: list[tuple[float, float, int]] = []
    orig_decide = RthCalculator.decide

    def spy(self, agent) -> str:
        pre = self.n_map_hits + self.n_map_fallbacks
        out = orig_decide(self, agent)
        seen.append((agent.battery.level_j, agent._arm_level_j,
                     self.n_map_hits + self.n_map_fallbacks - pre))
        return out

    monkeypatch.setattr(RthCalculator, "decide", spy)

    eng = _engine(cfg_on)
    result = eng.run()
    assert eng.rth.map_decide_on
    assert result.outcome is not None  # ran to a terminal outcome

    # non-vacuous: the energy decide ran, and every decide consumed exactly
    # one map/fallback lookup (proves the 7a decision path, not just arming)
    assert len(seen) >= 1
    assert all(delta >= 1 for _lvl, _arm, delta in seen)
    assert eng.rth.n_map_hits > 0

    # the 7b invariant end-to-end: no decide ever ran above the caller's arm
    for level_j, arm_j, _delta in seen:
        assert level_j <= arm_j + 1e-9

    # launch-lifecycle cross-check: with lambda=0 and no redistribution every
    # S0->S1 launch is one S1_TRANSIT sojourn, and every launch must have
    # armed -- not just "some arm was recorded"
    from uav_swarm_sim.infrastructure.enums import AgentState

    s1_counts: dict[int, int] = {}
    for s in result.history.sojourns():
        if s.state is AgentState.S1_TRANSIT:
            s1_counts[s.agent_id] = s1_counts.get(s.agent_id, 0) + 1
    assert sum(s1_counts.values()) >= 1, "no launch ever happened"
    for aid, a in eng.fleet.agents.items():
        assert len(a.sortie_arms) == s1_counts.get(aid, 0)
        for k, (sortie_idx, arm_j) in enumerate(a.sortie_arms, start=1):
            assert sortie_idx == k
            assert math.isfinite(arm_j)
            assert arm_j > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
