"""EM-01 Stage 3 gate: the ``route`` sub-flag owns ALL Stage-3 behaviour change,
and map routing resolves THE boxing livelock the straight chords cannot.

Byte-identity: ``enabled=True, route=False`` must produce a byte-identical
RUN to ``enabled=False`` (pins Stage 3's own flag the way test_energy_map_stage2
pins ``decide``; the stage-1/2 gates keep covering the flag-off path against
main). The check compares the full run signature -- the summary metrics AND
the complete FSM sojourn trajectory (every state, transition time and exit
reason) AND the outcome / coverage / stalled set -- not just a metrics digest,
so a Stage-3 regression cannot pass by leaving the summary metrics coincidentally
equal. Config hashes are deliberately NOT compared: flipping ``enabled`` on
changes the raw YAML and therefore the provenance hash by design; the byte-
identity contract is over the run OUTPUT. The default config's hash stays
unchanged structurally (``route`` is absent from default.yaml) -- that optional-
key contract is owned by the stage-1/2 gates.

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


def _run_signature(res) -> tuple:
    """The complete byte-identity artifact for a run: summary metrics + the
    full FSM sojourn trajectory (Sojourn is a frozen dataclass, so the lists
    compare by value) + the terminal outcome / coverage / stalled set."""
    return (
        _metrics_tuple(res.metrics),
        tuple(res.history.sojourns()),
        res.outcome,
        res.coverage_frac,
        res.stalled_agents,
    )


@pytest.mark.slow
def test_route_off_byte_identical(config_path):
    """enabled=True with route left at its default (False) consumes nothing --
    the full run (metrics + trajectory + outcome) is byte-identical."""
    base = _tiny_cfg(config_path)
    cfg_build = dataclasses.replace(
        base, rth=dataclasses.replace(base.rth, energy_map=EnergyMapConfig(enabled=True)))
    assert cfg_build.rth.energy_map.route is False  # guards the shipped default

    sig_off = _run_signature(_engine(base).run())
    sig_build = _run_signature(_engine(cfg_build).run())
    assert sig_build == sig_off


def _swaps_per_drone(res) -> dict[int, int]:
    counts: dict[int, int] = {}
    for s in res.history.sojourns():
        if s.state is AgentState.S_SWAP:
            counts[s.agent_id] = counts.get(s.agent_id, 0) + 1
    return counts


@pytest.mark.slow
def test_boxing_map_routing_rejects_buffered_resume():
    """EXP-09: the historical ring start is inside clearance, not a valid route.

    Previously the map resumed through that buffer and this test expected
    SUCCESS. Rejecting the unsafe fallback is an intentional bug fix; valid
    detours are exercised by test_exp09_coherent and the Stage-3 seam tests.
    """
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
    eng._build()
    assert eng.rth.map_route_on
    from shapely.geometry import Point
    from uav_swarm_sim.planning.visibility_router import RouteUnavailable
    unsafe = [a for a in eng.fleet.agents.values()
              if eng.env.buffered_obstacles.contains(Point(a.base.as_xy()))]
    assert unsafe
    for a in unsafe:
        assert not eng.env.in_obstacle(a.base.as_xy())
        with pytest.raises(RouteUnavailable, match="resume_blocked"):
            a._resume_transit()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
