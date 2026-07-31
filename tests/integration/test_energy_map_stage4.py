"""EM-01 Stage 4 gate: ``safety.stall_skip = False`` (the shipped default)
must reproduce the exact pre-Stage-4 ``main`` behaviour -- a cross-commit
golden, not the in-process A/B the Stage 1-3 gates use.

Why cross-commit. The Stage 1-3 gates compare two configs in the SAME
process: parent flag ON with the sub-flag OFF, versus everything OFF
(``test_energy_map_stage3.test_route_off_byte_identical`` etc.). Stage 4 has
no such parent -- ``stall_skip`` requires ``stall_detector``, whose
OFF-semantics are pinned by a DIFFERENT test
(``test_transit_livelock.test_fix_b4_stall_detector_cuts_the_livelock_early``).
An in-process A/B therefore cannot prove that Stage 4's own code
(``simulation_engine.py``'s ``SWAP_REQUEST`` routing, at the time of writing
around :442-446) left the flag-off path untouched -- both arms would run
through the SAME post-Stage-4 binary. The honest mechanism is a golden
captured on the last pre-Stage-4 commit and replayed against today's code.

Source commit: ``1bacad9`` (parent of the Stage 4 merge ``da76b7e``; verified
via ``git log --format="%h %p %s"``). On that commit ``SafetyConfig`` has no
``stall_skip`` field at all (Stage 4 introduced it), so the golden-generation
config is built via ``dataclasses.replace`` without that key -- the same
pattern ``test_transit_livelock._cfg`` already uses.

Regeneration procedure (both cases), if this golden ever needs to be redone:
    1. ``git worktree add <path> 1bacad9``
    2. Prepend ``<path>/src`` to ``sys.path`` (shadow the editable install)
       and run the run/canonicalise steps below against THAT commit's
       ``SimulationEngine`` -- same ``_run_signature`` /
       ``_canonical_signature_string`` / ``_golden_hash`` functions as here.
    3. Record the printed values as the ``_CASE_A_SIGNATURE`` tuple and the
       ``_CASE_B_*`` constants below.
    4. ``git worktree remove <path>``

Golden storage. Case (a)'s full run signature (metrics tuple + complete FSM
sojourn trajectory + outcome/coverage/stalled) is small (22 sojourns) and is
pinned as a literal tuple, exactly like the Stage 1-3 gates. Case (b) (the
FIX-B4 stall-cut replication) has 376 sojourns -- a literal tuple would be an
unreviewable ~40 KB blob -- so it is pinned as a SHA-256 hash of a CANONICAL
STRING built from explicitly named ``Sojourn`` fields (never ``repr()``,
which would couple the golden to ``__repr__`` and turn a cosmetic dataclass
change into a false physics-regression signal), plus a handful of
human-readable fields (outcome, coverage_frac, stalled_agents, skipped_legs,
n_swaps, duration_s, sojourn count) asserted separately so a mismatch is
diagnosable without decoding a hash.

Why replication 1 of study01_demand (not the M0 energy-map residual,
replication 7, used by ``test_stall_skip.test_stall_skip_turns_the_boxed_
replication_partial``): that replication requires ``rth.energy_map`` enabled,
and Task B1 (battery-zone demotion) is scheduled to change the map-ON path's
``critical_battery`` guard. A map-ON golden captured today would be
legitimately invalidated by the very next task, proving nothing about Stage
4 itself. Replication 1 with the map left at its config default (OFF) is
insulated from B1 and, via ``test_transit_livelock.test_fix_b4_stall_
detector_cuts_the_livelock_early``, is already provably known to hit the
stall budget (``stalled_agents == (3,)``) -- exactly the branch Stage 4
modified.

Expiry condition: this golden is a one-time Stage-4 identity proof. Any
DELIBERATE change to the map-OFF / stall_skip-OFF execution path invalidates
it and requires regeneration per the procedure above; that is a feature, not
a bug -- the test exists to catch ACCIDENTAL changes to that path.
"""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from uav_swarm_sim.infrastructure.config import load_config
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
# signature / canonicalisation helpers (identical on both sides of the       #
# golden -- any change here invalidates the pinned hash below)                #
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


def _canonical_sojourn(s) -> str:
    return f"{s.agent_id}|{s.state.value}|{s.t_in!r}|{s.t_out!r}|{s.reason_out}"


def _canonical_signature_string(res) -> str:
    m = res.metrics
    parts = [
        f"total_energy_j={m.total_energy_j!r}",
        f"duration_s={m.duration_s!r}",
        f"workload_std_m={m.workload_std_m!r}",
        f"n_swaps={m.n_swaps!r}",
        "per_agent_energy_j=" + ",".join(f"{k}:{v!r}" for k, v in sorted(m.per_agent_energy_j.items())),
        "per_agent_length_m=" + ",".join(f"{k}:{v!r}" for k, v in sorted(m.per_agent_length_m.items())),
        "sojourns=" + ";".join(_canonical_sojourn(s) for s in res.history.sojourns()),
        f"outcome={res.outcome.value}",
        f"coverage_frac={res.coverage_frac!r}",
        f"stalled_agents={getattr(res, 'stalled_agents', ())!r}",
        f"skipped_legs={getattr(res, 'skipped_legs', ())!r}",
    ]
    return "\n".join(parts)


def _golden_hash(res) -> str:
    return hashlib.sha256(_canonical_signature_string(res).encode()).hexdigest()


def _tiny_cfg(config_path):
    """Mirrors test_energy_map_stage3._tiny_cfg exactly (same fixture, same
    overrides) so case (a) is directly comparable to the Stage 1-3 gates."""
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


# --------------------------------------------------------------------------- #
# GOLDEN (a): tiny ordinary mission, captured on commit 1bacad9              #
# --------------------------------------------------------------------------- #
_CASE_A_SIGNATURE = (
    (110988.0, 266.0, 43.86164397170569, 0,
     {0: 54756.0, 1: 56232.0}, {0: 1688.2280307254143, 1: 1775.9513186688257}),
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
        Sojourn(agent_id=0, state=AgentState.S3_RTH, t_in=228.0, t_out=259.0, reason_out="mission_done"),
        Sojourn(agent_id=1, state=AgentState.S3_RTH, t_in=229.0, t_out=266.0, reason_out="mission_done"),
        Sojourn(agent_id=0, state=AgentState.S0_IDLE, t_in=259.0, t_out=266.0, reason_out="mission_end"),
        Sojourn(agent_id=1, state=AgentState.S0_IDLE, t_in=266.0, t_out=266.0, reason_out="mission_end"),
    ),
    Outcome.MISSION_SUCCESS,
    1.0,
    (),
)


@pytest.mark.slow
def test_stall_skip_off_ordinary_run_byte_identical_to_pre_stage4(config_path):
    """Case (a): the ordinary, non-stalling tiny mission. safety.stall_skip
    explicit-False must reproduce the exact pre-Stage-4 (1bacad9) run --
    summary metrics, the full FSM sojourn trajectory, outcome, coverage and
    the stalled set all match."""
    cfg = _tiny_cfg(config_path)
    cfg = dataclasses.replace(cfg, safety=dataclasses.replace(cfg.safety, stall_skip=False))
    res = _engine(cfg).run()
    assert _run_signature(res) == _CASE_A_SIGNATURE
    assert res.skipped_legs == ()


# --------------------------------------------------------------------------- #
# GOLDEN (b): study01_demand replication 1, FIX-B4 stall-cut, captured on    #
# commit 1bacad9. See the module docstring for why this replication (and    #
# not the M0 energy-map residual, replication 7) is the golden fixture.      #
# --------------------------------------------------------------------------- #
_CASE_B_OUTCOME = Outcome.MISSION_INCOMPLETE
_CASE_B_COVERAGE_FRAC = 0.7829312606630419
_CASE_B_STALLED_AGENTS = (3,)
_CASE_B_N_SWAPS = 10
_CASE_B_DURATION_S = 2016.5
_CASE_B_N_SOJOURNS = 376
_CASE_B_GOLDEN_HASH = "9fcd26b81ee210e252a535f2d9a7b9d55acd34460dace1395d468a247332477e"


@pytest.mark.slow
def test_stall_skip_off_livelock_replication_byte_identical_to_pre_stage4():
    """Case (b): the exact config from test_transit_livelock._cfg
    (transit_free_space=False, stall_detector=True), replication 1, PLUS
    safety.stall_skip explicit-False. Must reproduce the pre-Stage-4
    (1bacad9) run byte-for-byte: the human-readable fields for fast
    diagnosis, and the full canonical-signature hash for the complete FSM
    trajectory."""
    cfg = load_config("config/study01_demand.yaml")
    cfg = dataclasses.replace(
        cfg,
        fleet=dataclasses.replace(cfg.fleet, total_reserve_batteries=None),
        coverage=dataclasses.replace(cfg.coverage, transit_free_space=False),
        safety=dataclasses.replace(cfg.safety, stall_detector=True, stall_skip=False),
    )
    rng = RngFactory(cfg.sim.master_seed)
    res = SimulationEngine(cfg, rng, replication=1, planner=PlannerKind.DUBINS).run()

    assert res.outcome is _CASE_B_OUTCOME
    assert res.coverage_frac == _CASE_B_COVERAGE_FRAC
    assert res.stalled_agents == _CASE_B_STALLED_AGENTS
    assert res.skipped_legs == ()
    assert res.metrics.n_swaps == _CASE_B_N_SWAPS
    assert res.metrics.duration_s == _CASE_B_DURATION_S
    assert len(res.history.sojourns()) == _CASE_B_N_SOJOURNS
    assert _golden_hash(res) == _CASE_B_GOLDEN_HASH


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
