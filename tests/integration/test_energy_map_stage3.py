"""EM-01 Stage 3 gate: the ``route`` sub-flag owns ALL Stage-3 behaviour change,
and map routing resolves THE boxing livelock the straight chords cannot.

Byte-identity: ``enabled=True, route=False`` must stay metrics-byte-identical
to ``enabled=False`` (pins Stage 3's own flag the way test_energy_map_stage2
pins ``decide``; the stage-1/2 gates keep covering the flag-off path against
main).

THE BOXING TEST (the stage's acceptance criterion): the REAL pathological
replication from the transit-livelock root-cause report -- study01_demand
replication 1, drone 3, whose base -> leg[30].start resume chord crosses two
obstacle prisms. With B1 routing OFF that replication is a livelock
(MISSION_INCOMPLETE, stalled_agents == (3,), pinned by
test_transit_livelock.test_fix_b4_stall_detector_cuts_the_livelock_early --
deliberately not re-run here). Stage 3's claim: energy-map ROUTING ALONE
(``route=True``, ``decide=False``, B1 still OFF) turns that same replication
into MISSION_SUCCESS with a finite swap demand, because the resume transit
follows the map's parent-pointer polyline around the prisms instead of
replaying the blocked chord.
"""
from __future__ import annotations

import dataclasses

import pytest

from uav_swarm_sim.infrastructure.config import EnergyMapConfig, load_config
from uav_swarm_sim.infrastructure.enums import (
    AgentState,
    DecompositionAlgo,
    Outcome,
    PlannerKind,
)
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


def _engine(cfg, replication=0):
    return SimulationEngine(cfg, RngFactory(cfg.sim.master_seed),
                            replication=replication,
                            algo=DecompositionAlgo.TGC_BASIC,
                            planner=PlannerKind.DUBINS)


def _metrics_tuple(m):
    return (m.total_energy_j, m.duration_s, m.workload_std_m, m.n_swaps,
            dict(m.per_agent_energy_j), dict(m.per_agent_length_m))


@pytest.mark.slow
def test_route_off_metrics_byte_identical(config_path):
    """enabled=True with route left at its default (False) consumes nothing."""
    base = _tiny_cfg(config_path)
    cfg_build = dataclasses.replace(
        base, rth=dataclasses.replace(base.rth, energy_map=EnergyMapConfig(enabled=True)))
    assert cfg_build.rth.energy_map.route is False  # guards the shipped default

    m_off = _metrics_tuple(_engine(base).run().metrics)
    m_build = _metrics_tuple(_engine(cfg_build).run().metrics)
    assert m_build == m_off


def _swaps_per_drone(res) -> dict[int, int]:
    counts: dict[int, int] = {}
    for s in res.history.sojourns():
        if s.state is AgentState.S_SWAP:
            counts[s.agent_id] = counts.get(s.agent_id, 0) + 1
    return counts


@pytest.mark.slow
def test_boxing_map_routing_unblocks_the_livelocked_replication():
    """THE BOXING TEST: map routing alone (no B1, no map decide) resolves the
    replication that livelocks with straight chords."""
    cfg = load_config("config/study01_demand.yaml")
    cfg = dataclasses.replace(
        cfg,
        fleet=dataclasses.replace(cfg.fleet, total_reserve_batteries=None),
        coverage=dataclasses.replace(cfg.coverage, transit_free_space=False),
        safety=dataclasses.replace(cfg.safety, stall_detector=True),
        rth=dataclasses.replace(
            cfg.rth, energy_map=EnergyMapConfig(enabled=True, route=True)),
    )
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=1,
                           planner=PlannerKind.DUBINS)
    res = eng.run()
    assert eng.rth.map_route_on
    assert res.outcome is Outcome.MISSION_SUCCESS
    assert res.stalled_agents == ()
    swaps = _swaps_per_drone(res)
    # the pathological signature (151 swaps on drone #3) is gone: bounds mirror
    # the FIX-B1 arm of test_transit_livelock (healthy reps have D <= 11)
    assert sum(swaps.values()) <= 15
    assert swaps.get(3, 0) <= 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
