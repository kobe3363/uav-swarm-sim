"""Transit-livelock regression on the REAL pathological seed (FIX-B1 + FIX-B4).

Reference case (root-cause report, runs/spares_probe_demand): demand-mode
replication 1 of config/study01_demand.yaml (master_seed=42, unbounded pool)
livelocks -- drone #3 finishes coverage legs 0..29, its base->leg[30].start
resume chord crosses two obstacle prisms, and the S_OBS -> boxed-in -> RTH ->
unconditional-swap macro-loop burns 151 swaps until max_timesteps
(MISSION_INCOMPLETE, coverage 0.9081, battery >= 93 % throughout).

Both tests replay that exact replication end-to-end, so they are marked slow.
Flags are set via dataclasses.replace (not YAML), keeping them independent of
config/study01_demand.yaml's own flag values.

  * FIX-B1 on  -> the resume chord is routed around the prisms at plan time:
    the mission that livelocked now SUCCEEDS with a finite swap demand.
  * FIX-B4 on (B1 off) -> the livelock is cut after 5 no-progress swap cycles:
    early MISSION_INCOMPLETE with stalled_agents == (3,) instead of a
    max_timesteps burn (full flag-off byte-identity vs the GOLDEN batch is the
    experiment-level check; too slow for CI, see the delivery report).
"""
from __future__ import annotations

import dataclasses

import pytest

from uav_swarm_sim.infrastructure.config import load_config
from uav_swarm_sim.infrastructure.enums import AgentState, Outcome, PlannerKind
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import SimulationEngine

CONFIG = "config/study01_demand.yaml"
REPLICATION = 1
LOOP_DRONE = 3


def _cfg(transit_free_space: bool, stall_detector: bool):
    cfg = load_config(CONFIG)
    return dataclasses.replace(
        cfg,
        fleet=dataclasses.replace(cfg.fleet, total_reserve_batteries=None),
        coverage=dataclasses.replace(cfg.coverage, transit_free_space=transit_free_space),
        safety=dataclasses.replace(cfg.safety, stall_detector=stall_detector),
    )


def _run(cfg):
    rng = RngFactory(cfg.sim.master_seed)
    return SimulationEngine(cfg, rng, replication=REPLICATION,
                            planner=PlannerKind.DUBINS).run()


def _swaps_per_drone(res) -> dict[int, int]:
    counts: dict[int, int] = {}
    for s in res.history.sojourns():
        if s.state is AgentState.S_SWAP:
            counts[s.agent_id] = counts.get(s.agent_id, 0) + 1
    return counts


@pytest.mark.slow
def test_fix_b1_routed_transit_unblocks_the_livelocked_replication():
    res = _run(_cfg(transit_free_space=True, stall_detector=True))
    assert res.outcome is Outcome.MISSION_SUCCESS
    assert res.stalled_agents == ()
    swaps = _swaps_per_drone(res)
    # the pathological signature (151 swaps on drone #3) is gone: every drone's
    # demand is a small finite count, consistent with the healthy reps (D <= 11)
    assert sum(swaps.values()) <= 15
    assert swaps.get(LOOP_DRONE, 0) <= 5


@pytest.mark.slow
def test_fix_b4_stall_detector_cuts_the_livelock_early():
    res = _run(_cfg(transit_free_space=False, stall_detector=True))
    assert res.outcome is Outcome.MISSION_INCOMPLETE
    assert res.stalled_agents == (LOOP_DRONE,)
    swaps = _swaps_per_drone(res)
    # cut after 5 consecutive no-progress swap cycles: the 151-swap burn cannot
    # happen (first swap + 5 no-progress cycles + the in-flight request)
    assert swaps[LOOP_DRONE] <= 8
