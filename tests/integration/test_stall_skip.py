"""EM-01 Stage 4 -- runtime skip-on-stall (``safety.stall_skip``).

Design pivot (author-approved): the proposal's seam 7e plan-time criterion
("coverage-leg entry cell has E_home=inf -> skip") was FALSIFIED by the Stage-4
M0 measurement -- 432/5520 inf-entry strips across all 50 study01_demand
replications fly successfully today (a 20 m red cell is not an unreachability
test: the reactive S_OBS machinery threads sub-cell corridors), so a plan-time
skip would have turned every SUCCESS replication PARTIAL. The shipped detection
seam is therefore RUNTIME: the FIX-B4 StallDetector's budget hit (5 no-progress
swap cycles -- the empirical proof the leg is unreachable for this executor)
forfeits the stuck leg instead of halting the run. Explicit accounting:
``MissionResult.skipped_legs`` + terminal ``Outcome.MISSION_PARTIAL`` + an
honestly reduced ``coverage_frac``.

A SEPARATE flag (not folded into stall_detector) because the FIX-B4 halt
semantics (stall -> early MISSION_INCOMPLETE + stalled_agents) are pinned by
test_transit_livelock.test_fix_b4_stall_detector_cuts_the_livelock_early; that
test keeps covering the skip-off path against main.

THE RESIDUAL TEST replays study01_demand replication 7 -- one of the 4/50 M0
residuals (drone #0 stall-cycles on coverage leg 48, whose entry cell is
in-grid with E_home=inf: a genuine obstacle-boxed pocket): with stall_skip on,
the replication that stall-halted as MISSION_INCOMPLETE now terminates cleanly
as MISSION_PARTIAL with the forfeited strip on the record.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from uav_swarm_sim.execution.agent import Agent
from uav_swarm_sim.execution.events import EventBus
from uav_swarm_sim.infrastructure.config import ConfigError, EnergyMapConfig, load_config
from uav_swarm_sim.infrastructure.core_types import Event
from uav_swarm_sim.infrastructure.enums import (
    EventType,
    MissionType,
    Outcome,
    PlannerKind,
)
from uav_swarm_sim.infrastructure.rng import RngFactory
from uav_swarm_sim.infrastructure.simulation_engine import (
    _STALL_SWAP_BUDGET,
    SimulationEngine,
    StallDetector,
)


# --------------------------------------------------------------------------- #
# flag surface                                                                 #
# --------------------------------------------------------------------------- #
def test_stall_skip_defaults_off(config_path):
    cfg = load_config(config_path)
    assert cfg.safety.stall_skip is False


def test_stall_skip_requires_the_detector(config_path):
    with pytest.raises(ConfigError, match="stall_skip requires"):
        load_config(config_path, overrides={"safety.stall_skip": True})


# --------------------------------------------------------------------------- #
# Agent.skip_stuck_leg unit surface (duck-typed stub: the method touches only  #
# the coverage-progress attributes)                                            #
# --------------------------------------------------------------------------- #
class _LegStub:
    def __init__(self, n_legs: int, cov_idx: int) -> None:
        self._cov_legs = [object()] * n_legs
        self._cov_idx = cov_idx
        self._skipped_cov: tuple[int, ...] = ()
        self._cov_frac_deficit = 0

    skip_stuck_leg = Agent.skip_stuck_leg


def test_skip_on_a_strip_forfeits_strip_and_following_connector():
    s = _LegStub(10, 4)
    Agent.skip_stuck_leg(s)
    assert s._cov_idx == 6
    assert s._skipped_cov == (4,)
    assert s._cov_frac_deficit == 2


def test_skip_on_a_connector_drops_it_without_coverage_loss():
    s = _LegStub(10, 5)
    Agent.skip_stuck_leg(s)
    assert s._cov_idx == 6
    assert s._skipped_cov == ()
    assert s._cov_frac_deficit == 0


def test_skip_on_the_last_strip_caps_at_the_leg_count():
    s = _LegStub(49, 48)
    Agent.skip_stuck_leg(s)
    assert s._cov_idx == 49
    assert s._skipped_cov == (48,)
    assert s._cov_frac_deficit == 1


def test_skip_past_the_end_is_a_noop():
    s = _LegStub(4, 4)
    Agent.skip_stuck_leg(s)
    assert s._cov_idx == 4
    assert s._skipped_cov == ()
    assert s._cov_frac_deficit == 0


def test_repeated_stalls_walk_the_plan_and_terminate():
    """Worst case (every strip unreachable): alternating strip/connector skips
    reach the end of a finite leg list -- no infinite skip loop."""
    s = _LegStub(9, 0)
    for _ in range(20):
        Agent.skip_stuck_leg(s)
    assert s._cov_idx == 9
    assert s._skipped_cov == (0, 2, 4, 6, 8)
    assert s._cov_frac_deficit == 9


# --------------------------------------------------------------------------- #
# the mission-type boundary: skip-on-stall is BOUSTROPHEDON ONLY               #
# --------------------------------------------------------------------------- #
class _EngineStub:
    """Duck-typed engine for ``SimulationEngine._route_events``: the
    SWAP_REQUEST branch reads only these five attributes, so the guard can be
    pinned without building (and flying) a whole mission."""

    def __init__(self, mission_type: MissionType) -> None:
        self.bus = EventBus()
        self._stall = StallDetector()
        self._stall_skip = True
        self._mission_type = mission_type
        self.agent = _LegStub(10, 4)
        self.fleet = SimpleNamespace(agents={3: self.agent})
        # underscore-prefixed: the stub mirrors SwapStation.request(aid, t) but
        # the queue itself is irrelevant to the guard under test
        self.swap_station = SimpleNamespace(request=lambda _aid, _t: None)

    def stall_out(self) -> None:
        """Route enough no-progress swap requests to hit the stall budget."""
        for _ in range(_STALL_SWAP_BUDGET + 1):
            self.bus.publish(Event(EventType.SWAP_REQUEST, 0.0, {"agent_id": 3}))
            SimulationEngine._route_events(self, 0.0)


def test_area_coverage_stall_skips_the_leg_and_clears_the_halt_flag():
    eng = _EngineStub(MissionType.COVERAGE)
    eng.stall_out()
    assert eng.agent._skipped_cov == (4,)
    assert eng.agent._cov_idx == 6
    assert eng._stall.stalled == set()      # un-flagged: the mission goes on


def test_target_visit_never_skips_and_keeps_the_fix_b4_halt():
    """Tour plans have no strip/connector structure and their coverage_frac
    would credit a skipped target as visited, so stall_skip is a deliberate
    no-op there: the agent keeps its leg and stays flagged, which is what the
    engine's early-halt turns into MISSION_INCOMPLETE (never MISSION_PARTIAL --
    _skipped_legs stays empty, so the PARTIAL branch cannot fire)."""
    eng = _EngineStub(MissionType.TARGET_VISIT)
    eng.stall_out()
    assert eng.agent._skipped_cov == ()
    assert eng.agent._cov_idx == 4          # leg untouched
    assert eng._stall.stalled == {3}        # FIX-B4 halt path intact


# --------------------------------------------------------------------------- #
# THE RESIDUAL TEST (M0 rep 7) -- the stage's acceptance criterion             #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_stall_skip_turns_the_boxed_replication_partial():
    cfg = load_config("config/study01_demand.yaml")
    cfg = dataclasses.replace(
        cfg,
        fleet=dataclasses.replace(cfg.fleet, total_reserve_batteries=None),
        coverage=dataclasses.replace(cfg.coverage, transit_free_space=False),
        safety=dataclasses.replace(cfg.safety, stall_detector=True, stall_skip=True),
        rth=dataclasses.replace(
            cfg.rth,
            energy_map=EnergyMapConfig(enabled=True, decide=True, route=True)),
    )
    eng = SimulationEngine(cfg, RngFactory(cfg.sim.master_seed), replication=7,
                           planner=PlannerKind.DUBINS)
    res = eng.run()
    # skipped-not-stalled: the forfeit is on the record, the halt flag is not
    assert res.outcome is Outcome.MISSION_PARTIAL
    assert res.skipped_legs == ((0, 48),)
    assert res.stalled_agents == ()
    # honest coverage: below 1.0 by exactly the forfeited strip's credit
    assert 0.99 <= res.coverage_frac < 1.0
    # clean termination, no timestep burn to the ceiling
    assert res.metrics.duration_s < 0.5 * cfg.sim.max_timesteps * cfg.sim.dt_s


# --------------------------------------------------------------------------- #
# author's guard: the demand success predicate treats PARTIAL as non-success   #
# --------------------------------------------------------------------------- #
def test_demand_predicate_counts_partial_as_infinite_demand(config_path, monkeypatch):
    """MISSION_PARTIAL must stay a non-success (D = infinity) in demand mode:
    the strict identity ``res.outcome is Outcome.MISSION_SUCCESS``
    (run_spare_sizing.run_demand) may never be widened to include PARTIAL
    without shifting the STUDY-01 demand CDF / success ceiling."""
    from uav_swarm_sim.experiments import run_spare_sizing
    from uav_swarm_sim.experiments.spare_sizing import demand_success_count

    class _PartialEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self):
            return SimpleNamespace(
                outcome=Outcome.MISSION_PARTIAL,
                metrics=SimpleNamespace(n_swaps=3),
                history=SimpleNamespace(sojourns=lambda: []),
            )

    monkeypatch.setattr(run_spare_sizing, "SimulationEngine", _PartialEngine)
    cfg = load_config(config_path)
    records = run_spare_sizing.run_demand(cfg, reps=1, rng=RngFactory(1))
    assert len(records) == 1
    assert records[0].outcome == "MISSION_PARTIAL"
    assert records[0].demand is None                      # D = infinity
    assert demand_success_count(records, 10**6) == 0      # at ANY pool size


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
