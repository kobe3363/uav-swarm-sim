"""EM-01 B1 gate: the ``zone_demotion`` sub-flag removes the static CRITICAL RTH
net (0.40) so the dynamic map governs the normal energy return, deepening the
usable sortie to the 0.20 TERMINAL floor.

Two gates:

(1) FLAG-OFF BYTE-IDENTITY -- a CROSS-COMMIT golden, not the in-process A/B the
Stage 1-3 gates use. ``zone_demotion`` only changes behaviour when True, so an
in-process ``zone_demotion=False`` vs no-flag comparison would be a self-
comparison in one post-B1 binary (proves nothing). Instead a full run signature
is captured on the PRE-B1 base commit (``0409186``, the HEAD this change branches
from) for a tiny map-ON ``decide+route`` mission and pinned as a literal tuple.
That mission is deliberately one that NEVER crosses CRITICAL (it completes on a
healthy battery -- 22 sojourns, all ``coverage_complete``), so the guard edit is
exercised-as-inert: post-B1, the same run with ``zone_demotion`` at its default
must reproduce the golden byte-for-byte. The helpers below are REPLICATED from
test_energy_map_stage4 (case a) rather than imported -- each gate owns its own
signature machinery so a refactor of one cannot silently move another's goalposts.

(2) THE OBSERVABLE (B1 DoD) -- on the operational study01_demand config with the
map deciding, ``zone_demotion=True`` shifts the return-reason attribution: the
static ``critical_battery`` net drops to 0 and the dynamic ``rth_energy`` decision
takes over. Asserted as a DIRECTION (measured 6-rep A/B: critical 49->0,
rth_energy 0->16), not a hand-tuned count -- swap/return demand is physics-
dependent and exact pinning would break on later flag-on stages.
"""
from __future__ import annotations

import dataclasses
from collections import Counter

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
from uav_swarm_sim.metrics.state_history import Sojourn


# --------------------------------------------------------------------------- #
# signature helpers (REPLICATED from test_energy_map_stage4 case a)          #
# --------------------------------------------------------------------------- #
def _metrics_tuple(m):
    return (m.total_energy_j, m.duration_s, m.workload_std_m, m.n_swaps,
            dict(m.per_agent_energy_j), dict(m.per_agent_length_m))


def _run_signature(res) -> tuple:
    return (
        _metrics_tuple(res.metrics),
        tuple(res.history.sojourns()),
        res.outcome,
        res.coverage_frac,
        getattr(res, "stalled_agents", ()),
    )


def _tiny_cfg(config_path):
    """Mirrors test_energy_map_stage3/4._tiny_cfg exactly, then turns the map ON
    (enabled+decide+route) with zone_demotion left at its default (False)."""
    cfg = load_config(str(config_path), overrides={
        "fleet.n_drones": 2,
        "fleet.battery_capacity_wh": 400.0,
        "failure.hazard_rate_per_hour": 0.0,
        "env.geojson_path": "data/areas/smoke_area.geojson",
        "env.obstacle_density_per_km2": 4.0,
        "sim.dt_s": 1.0,
        "sim.max_timesteps": 20000,
        "telemetry.enabled": False,
    })
    return dataclasses.replace(
        cfg, rth=dataclasses.replace(
            cfg.rth, energy_map=EnergyMapConfig(enabled=True, decide=True, route=True)))


def _engine(cfg, replication=0):
    return SimulationEngine(cfg, RngFactory(cfg.sim.master_seed),
                            replication=replication,
                            algo=DecompositionAlgo.TGC_BASIC,
                            planner=PlannerKind.DUBINS)


# --------------------------------------------------------------------------- #
# GOLDEN: tiny map-ON (decide+route) mission, captured on PRE-B1 commit       #
# 0409186. zone_demotion defaults False and this mission never crosses        #
# CRITICAL, so post-B1 the run must reproduce this signature byte-for-byte.    #
# --------------------------------------------------------------------------- #
_GOLDEN_SIGNATURE = (
    (113172.0, 272.0, 53.70099459505229, 0,
     {0: 55640.0, 1: 57532.0}, {0: 1709.538321254871, 1: 1816.9403104449757}),
    (
        Sojourn(agent_id=0, state=AgentState.S0_IDLE, t_in=0.0, t_out=0.0, reason_out="launch"),
        Sojourn(agent_id=1, state=AgentState.S0_IDLE, t_in=0.0, t_out=0.0, reason_out="launch"),
        Sojourn(agent_id=0, state=AgentState.S1_TRANSIT, t_in=0.0, t_out=4.0, reason_out="zone_entry"),
        Sojourn(agent_id=1, state=AgentState.S1_TRANSIT, t_in=0.0, t_out=5.0, reason_out="zone_entry"),
        Sojourn(agent_id=0, state=AgentState.S2_MISSION, t_in=4.0, t_out=54.0, reason_out="ferry_start"),
        Sojourn(agent_id=1, state=AgentState.S2_MISSION, t_in=5.0, t_out=55.0, reason_out="ferry_start"),
        Sojourn(agent_id=0, state=AgentState.S_FERRY, t_in=54.0, t_out=62.0, reason_out="ferry_end"),
        Sojourn(agent_id=1, state=AgentState.S_FERRY, t_in=55.0, t_out=63.0, reason_out="ferry_end"),
        Sojourn(agent_id=0, state=AgentState.S2_MISSION, t_in=62.0, t_out=112.0, reason_out="ferry_start"),
        Sojourn(agent_id=1, state=AgentState.S2_MISSION, t_in=63.0, t_out=113.0, reason_out="ferry_start"),
        Sojourn(agent_id=0, state=AgentState.S_FERRY, t_in=112.0, t_out=120.0, reason_out="ferry_end"),
        Sojourn(agent_id=1, state=AgentState.S_FERRY, t_in=113.0, t_out=121.0, reason_out="ferry_end"),
        Sojourn(agent_id=0, state=AgentState.S2_MISSION, t_in=120.0, t_out=170.0, reason_out="ferry_start"),
        Sojourn(agent_id=1, state=AgentState.S2_MISSION, t_in=121.0, t_out=171.0, reason_out="ferry_start"),
        Sojourn(agent_id=0, state=AgentState.S_FERRY, t_in=170.0, t_out=178.0, reason_out="ferry_end"),
        Sojourn(agent_id=1, state=AgentState.S_FERRY, t_in=171.0, t_out=179.0, reason_out="ferry_end"),
        Sojourn(agent_id=0, state=AgentState.S2_MISSION, t_in=178.0, t_out=228.0, reason_out="coverage_complete"),
        Sojourn(agent_id=1, state=AgentState.S2_MISSION, t_in=179.0, t_out=229.0, reason_out="coverage_complete"),
        Sojourn(agent_id=0, state=AgentState.S3_RTH, t_in=228.0, t_out=263.0, reason_out="mission_done"),
        Sojourn(agent_id=1, state=AgentState.S3_RTH, t_in=229.0, t_out=272.0, reason_out="mission_done"),
        Sojourn(agent_id=0, state=AgentState.S0_IDLE, t_in=263.0, t_out=272.0, reason_out="mission_end"),
        Sojourn(agent_id=1, state=AgentState.S0_IDLE, t_in=272.0, t_out=272.0, reason_out="mission_end"),
    ),
    Outcome.MISSION_SUCCESS,
    1.0,
    (),
)


@pytest.mark.slow
def test_zone_demotion_off_byte_identical_to_pre_b1(config_path):
    """Gate (1): map ON (decide+route) with zone_demotion at its default (False)
    reproduces the exact pre-B1 (0409186) run -- summary metrics, the full FSM
    sojourn trajectory, outcome, coverage and the stalled set all match. The
    mission never crosses CRITICAL, so the demoted guard branch is inert here and
    the signature pins that the B1 edit did not perturb the default path."""
    cfg = _tiny_cfg(config_path)
    assert cfg.rth.energy_map.zone_demotion is False  # guards the shipped default
    res = _engine(cfg).run()
    assert _run_signature(res) == _GOLDEN_SIGNATURE
    assert res.skipped_legs == ()


def _return_reason_counts(res) -> Counter:
    """Counts of the return-attribution reasons over S2_MISSION / S_FERRY exits."""
    c: Counter = Counter()
    for s in res.history.sojourns():
        if s.state in (AgentState.S2_MISSION, AgentState.S_FERRY):
            c[s.reason_out] += 1
    return c


def _study01_engine(zone_demotion: bool):
    cfg = load_config("config/study01_demand.yaml")
    cfg = dataclasses.replace(cfg, rth=dataclasses.replace(
        cfg.rth, energy_map=EnergyMapConfig(
            enabled=True, decide=True, route=True, zone_demotion=zone_demotion)))
    return SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=1,
                            planner=PlannerKind.DUBINS)


@pytest.mark.slow
def test_zone_demotion_shifts_return_attribution_to_rth_energy(config_path):
    """Gate (2), the B1 observable: on the operational study01_demand config with
    the map deciding, turning zone_demotion ON removes the static CRITICAL net so
    the dynamic map governs the return. One replication is enough to show the
    DIRECTION (the 6-rep A/B measured critical 49->0, rth_energy 0->16); exact
    counts are physics-dependent and deliberately NOT pinned."""
    off = _return_reason_counts(_study01_engine(zone_demotion=False).run())
    on = _return_reason_counts(_study01_engine(zone_demotion=True).run())

    # OFF: the static net governs -> critical_battery present, rth_energy absent
    assert off["critical_battery"] > 0
    assert off["rth_energy"] == 0
    # ON: the static net is gone -> critical_battery eliminated, rth_energy takes over
    assert on["critical_battery"] == 0
    assert on["rth_energy"] > off["rth_energy"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
